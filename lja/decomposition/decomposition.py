import glob, os
import numpy as np
from sklearn.utils import extmath
import warnings
import matplotlib.pyplot as plt
from lja.analyser.plotter import Plotter


class Decomposition:
    """Creates an Decomposition object, that calculates singular vectors using regularized, randomized SVD."""

    def __init__(self, path, show_plots=False):
        super(Decomposition, self).__init__()

        self.plotter = Plotter(path, show_plots)
        self.path = path
        self.transformations = []
        self.activations = []
        self.decompositions = []
        self.number_of_layers = None
        self.side = None
        self.labels = None

    def load(self):

        # path
        path = "results/transformations/" + self.path
        print("Load from: ", path)

        # config
        self.number_of_layers = len(next(os.walk(path))[1])

        # loop through layer folders
        for layer in range(self.number_of_layers):

            path_layer = path + "/Layer" + str(layer) + "/"
            self.activations.append(np.load(path_layer + "activation.npy"))

            if layer < self.number_of_layers - 1:
                self.transformations.append(np.load(path_layer + "transformation.npy"))

        # load labels
        self.labels = np.load(path + "labels.npy")

        pass

    def store(self):

        # path
        path = "results/decompositions/" + self.path + self.side + "/"
        print("\nStore in: ", path)

        # loop through layers
        for layer, decomposition in enumerate(self.decompositions):

            # create folder
            path_layer = path + "Layer" + str(layer) + "/"
            if not os.path.exists(path_layer):
                os.makedirs(path_layer)

            # store
            for item, name in zip(decomposition, ["u", "s", "vh", "k"]):
                np.save(path_layer + name + ".npy", item)

        pass

    def decompose(self, k_list, side="left"):

        # reset decompositions
        self.decompositions = []
        self.side = side

        # loop through layers
        for layer, T in enumerate(self.transformations):
            print("\nLayer:", layer)

            # obtain decomposition
            u, s, vh, k = self.get_decomposition(T, k_list[layer], self.side)

            # store
            self.decompositions.append((u, s, vh, k))

            # test
            self.test_decomposition(
                u, s, vh, self.activations[layer], self.activations[layer + 1], k
            )

        pass

    def test_decomposition(self, u, s, vh, x, xp1, k):

        print(
            "Stacked side:\t",
            self.side,
            "\nK:\t\t",
            k,
            "\nSize of u:\t",
            u.shape,
            "\nSize of s\t",
            s.shape,
            "\nSize of vh\t",
            vh.shape,
        )

        # deviation tolerance
        tol = 1e-3

        if self.side == "left":
            U = u

            # 1. What is read:
            x_ext = np.insert(x, x.shape[1], 1, axis=1)
            read_in = vh @ x_ext.T

            # 2. Scale
            read_in_scaled = np.diag(s) @ read_in

            # 3. Write to ouput
            t = U @ read_in_scaled
            xp1_hat = t[range(t.shape[0]), :, range(t.shape[0])]

        elif self.side == "right":
            VH = vh

            # 1. What is read:
            x_ext = np.insert(x, x.shape[1], 1, axis=1)
            read_ins = VH @ x_ext.T

            # pick first column of first read_in and so on
            read_in = np.transpose(
                read_ins[range(read_ins.shape[0]), :, range(read_ins.shape[0])]
            )

            # 2. Scale
            read_in_scaled = np.diag(s) @ read_in

            # 3. Write to output
            xp1_hat = np.transpose(u @ read_in_scaled)

        # 4. Compare to actual output
        acc = np.sum(np.isclose(xp1_hat, xp1, atol=tol)) / xp1_hat.size
        error = np.mean(np.abs(xp1_hat - xp1))

        print(f"Reconstruction accuracy (tol={tol}):", acc)
        print("Reconstruction AbsError:", error)

        return acc, error

    def get_decomposition(self, T, k, side):

        if side == "left":

            # 1. stack transformations of each input
            T_stacked = np.vstack(T)

            # 2. Apply SVD
            if T_stacked.shape[0] < k:
                k = T_stacked.shape[0]
                warnings.warn(
                    "k has been automatically reduced - k larger than stacked datapoints available -  increase n to avoid this"
                )

            elif T.shape[2] < k:
                k = T.shape[2]
                warnings.warn(
                    "k has been automatically reduced - k larger than available dimensions"
                )

            # apply svd
            u_stacked, s, vh = extmath.randomized_svd(T_stacked, k, random_state=1)

            # 3. Recover single U Matrices
            U = u_stacked.reshape(T.shape[0], T.shape[1], k)

            # test if sorted
            # print(all(s[i] >= s[i + 1] for i in range(len(s) - 1)))

            return U, s, vh, k

        elif side == "right":

            # 1. stack transformations of each input
            T_stacked = np.hstack(T)

            # 2. Apply SVD
            k = min(k, T.shape[1])
            u, s, vh_stacked = extmath.randomized_svd(
                T_stacked, k, random_state=1, n_oversamples=100
            )

            # 3. Recover single VH Matices
            vh_stacked = vh_stacked.reshape(k, T.shape[0], T.shape[2])
            VH = np.hstack(vh_stacked).reshape(T.shape[0], k, T.shape[2])

            return u, s, VH, k

        else:
            raise Exception("Decomposition: invalid side")

    def get_decomposition_by_layer_index(self, layer, k, side="left", test=False):

        # config
        self.side = side

        if layer >= len(self.transformations):
            raise Exception("Decomposition: invalid layer index")

        else:
            u, s, vh, k = self.get_decomposition(self.transformations[layer], k, side)

            # test
            if test:
                self.test_decomposition(
                    u, s, vh, self.activations[layer], self.activations[layer + 1], k
                )

            return u, s, vh, k

    def get_error_function(self, layer, k_range, side="left"):

        # memory
        errors = []
        accuracies = []

        for k in k_range:

            print("\n-- K:", k)

            # obtain decomposition
            u, s, vh, k = self.get_decomposition_by_layer_index(layer, k, side="left")

            acc, error = self.test_decomposition(
                u, s, vh, self.activations[layer], self.activations[layer + 1], k
            )

            # store
            errors.append(error)
            accuracies.append(acc)

        # plot
        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_line_plot(
            k_range,
            errors,
            "Mean absolute error",
            "MAE",
            "Number of components",
            "reconstruction_error",
        )
        self.plotter.plot_line_plot(
            k_range,
            accuracies,
            "Reconstruction Accuracy",
            "Accuracy",
            "Number of components",
            "reconstruction_accuracy",
        )

        pass
