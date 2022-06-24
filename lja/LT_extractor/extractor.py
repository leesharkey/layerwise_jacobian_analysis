import os.path
import torch
import numpy as np


class LTExtractor:
    """Creates an Extractor object, that calculates the jacobians of a given entwork."""

    def __init__(self, net, x0, labels):
        super(LTExtractor, self).__init__()

        self.net = net
        self.x0 = x0
        self.activations = [x0]
        self.linear_transformations = []
        self.results_path = "results/transformations/"
        self.labels = labels

    def store(self, end_path):

        # root folder
        path = self.results_path + end_path
        print("Store LT in:", path)

        # loop through layer folders
        for layer in range(len(self.activations)):

            # create folder
            path_layer = path + "Layer" + str(layer) + "/"
            if not os.path.exists(path_layer):
                os.makedirs(path_layer)

            np.save(
                path_layer + "activation.npy",
                self.activations[layer].detach().cpu().numpy(),
            )

            if layer < len(self.linear_transformations):
                np.save(
                    path_layer + "transformation.npy",
                    self.linear_transformations[layer].detach().cpu().numpy(),
                )

        # save labels
        np.save(
            path + "labels.npy", self.labels.detach().cpu().numpy(),
        )

        pass

    def extract(self):
        # reset
        self.activations = [self.x0]
        self.linear_transformations = []

        # loop through all layers
        for i in range(self.net.get_depth()):

            # get layer
            layer, act = self.net.get_layer_and_act(i)
            print("\n\nIndex :", i, "\nLayer:", layer, "\nAct:", act)

            # obtain linear transfromations for that layer
            activation, linear_transformation = self.get_linear_transformation(
                layer, act, self.activations[i]
            )

            # store
            self.activations.append(activation)
            self.linear_transformations.append(linear_transformation)

        pass

    def get_linear_transformation(self, layer, act, x):

        # 1. Pre-activation and activation
        preact = layer(x)
        y = act(preact)

        # 2. parameters
        params = layer.weight.data
        bias = layer.bias.data

        # 3. Extend for bias
        params_ext = torch.cat((params, bias[:, None]), dim=1)

        # 2. Compute scaling vector
        scaling_vec = y / preact
        scaling_vec[scaling_vec != scaling_vec] = 0

        # 3. Apply scaling on parameters
        scaling_vec = scaling_vec[:, :, None].expand(-1, -1, params_ext.shape[1])
        linear_tranformations_ext = torch.multiply(params_ext, scaling_vec)

        self.test_transformation(x, y, linear_tranformations_ext, params_ext)

        return y, linear_tranformations_ext

    def test_transformation(self, x, y, linear_tranformations_ext, params_ext):

        # dimsenions
        n = x.shape[0]
        dim_input = params_ext.shape[1]
        device = params_ext.device

        # Extend x with bias
        x_ext = torch.cat((x, torch.ones(n, 1, device=device)), dim=-1)
        x_ext = x_ext[:, None, :].reshape((n, dim_input, 1))

        # Claculate y_prime
        y_prime = torch.squeeze(torch.bmm(linear_tranformations_ext, x_ext))

        self.compare_transformations(linear_tranformations_ext, params_ext, y_prime, y)

        pass

    def compare_transformations(self, linear_tranformations, parameter, y_prime, y):

        # comaprison
        num_matches = torch.isclose(linear_tranformations, parameter, atol=1e-2).sum(
            dim=(1, 2)
        )
        frac_match = num_matches / torch.numel(linear_tranformations[0])
        diff = torch.abs(y - y_prime).sum(dim=1)

        print("\nCheck params:")
        print("Min Fraction matched in weight and LT matrix:", frac_match.min().item())
        print("Max Fraction matched in weight and LT matrix:", frac_match.max().item())
        print(
            "Average Fraction matched in weight and LT matrix: %f" % frac_match.mean()
        )

        print("\nCheck predcition:")
        print("Max Difference between y and y_prime::", diff.max().item())
        print("Mean Difference  between y and y_prime: %f" % diff.mean())
        print()

        pass
