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

    def get_decomposition_by_layer_index(self, layer, k, side="left"):

        # config
        self.side = side

        if layer >= len(self.transformations):
            raise Exception("Decomposition: invalid layer index")

        else:
            u, s, vh, k = self.get_decomposition(self.transformations[layer], k, side)

            return u, s, vh, k
