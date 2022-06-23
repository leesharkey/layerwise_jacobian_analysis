import os.path

import torch
import numpy as np
from lja.LT_extractor.activation_derivatives import get_derivative


class LTExtractor:
    """Creates an Extractor object, that calculates the jacobians of a given entwork."""

    def __init__(self, net, x0, labels):
        super(LTExtractor, self).__init__()

        self.net = net
        self.activations = [x0]
        self.linear_transformations = []
        self.results_path = "results/transformations/"
        self.labels = labels

    def store(self, end_path):
        path = self.results_path + end_path
        print("Store LT in:", path)
        if not os.path.exists(path):
            os.makedirs(path)

        for i, activation in enumerate(self.activations):
            np.save(
                path + "activation_" + str(i) + ".npy",
                activation.detach().cpu().numpy(),
            )

        for i, transformation in enumerate(self.linear_transformations):
            np.save(
                path + "transformation_" + str(i) + ".npy",
                transformation.detach().cpu().numpy(),
            )

        np.save(
            path + "labels.npy", self.labels.detach().cpu().numpy(),
        )

    def extract(self):
        for i in range(self.net.get_depth()):
            layer, act = self.net.get_layer_and_act(i)
            print("\nIndex :", i, "\nLayer:", layer, "\nAct:", act)
            activation, linear_transformation = self.get_linear_transformation(
                layer, act, self.activations[i]
            )
            self.activations.append(activation)
            self.linear_transformations.append(linear_transformation)

    def get_linear_transformation(self, layer, act, x):

        # 1. Pre-activation and activation
        preact = layer(x)  # [n, dim_output]
        y = act(preact)  # [n, dim_output]

        # 2. Derivative
        derivative_function = get_derivative(act)
        derivative = derivative_function(preact)  # [n, dim_output]

        # 2. parameters
        params = layer.weight.data
        bias = layer.bias.data

        # 3. Extend for bias
        params_ext = torch.cat(
            (params, bias[:, None]), dim=1
        )  # [dim_output, dim_input]

        n = x.shape[0]
        dim_output = params_ext.shape[0]
        dim_input = params_ext.shape[1]
        derivative_ext = derivative[:, :, None].expand(
            -1, dim_output, dim_input
        )  # [n, dim_output, dim_input]

        # 4. Calculate linear matrix
        linear_tranformations_ext = torch.multiply(
            params_ext, derivative_ext
        )  # [n, dim_output, dim_input]

        # 5. Test
        self.test_transformation(x, y, linear_tranformations_ext, params_ext)

        return y, linear_tranformations_ext

    def test_transformation(self, x, y, linear_tranformations_ext, params_ext):

        # Extend x
        n = x.shape[0]
        dim_input = params_ext.shape[1]
        device = params_ext.device
        x_ext = torch.cat(
            (x, torch.ones(n, 1, device=device)), dim=-1
        )  # [n, dim_input]
        x_ext = x_ext[:, None, :].reshape((n, dim_input, 1))  # [n, dim_input, 1]

        # Claculate y_prime
        y_prime = torch.squeeze(
            torch.bmm(linear_tranformations_ext, x_ext)
        )  # [n, dim_output]

        self.compare_transformations(linear_tranformations_ext, params_ext, y_prime, y)

    def compare_transformations(self, linear_tranformations, parameter, y_prime, y):

        print("Check:")
        num_matches = torch.isclose(linear_tranformations, parameter, atol=1e-2).sum(
            dim=(1, 2)
        )
        frac_match = num_matches / torch.numel(linear_tranformations[0])
        print("Min Fraction matched:", frac_match.min().item())
        print("Max Fraction matched:", frac_match.max().item())
        print(
            "Average Fraction matched in linear tranformation matrices and weight matrix: %f"
            % frac_match.mean()
        )

        diff = torch.abs(y - y_prime).sum(dim=1)
        print("Max Differences:", diff.max().item())
        print("Mean diff between y and y_prime: %f" % diff.mean())
        print()
