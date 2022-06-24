import glob, os
import numpy as np


class Dataloader:
    """Creates an loader object, that can load all the data from the folder, transformations, decompositions and clusters."""

    def __init__(self, path):
        super(Dataloader, self).__init__()

        self.path = path

        # decomposition
        self.u_list = []
        self.vh_list = []
        self.s_list = []
        self.k_list = []

        # activations
        self.activation_list = []
        self.transformation_list = []
        self.labels = None

        # clusters
        self.clusters = []

        # general
        self.number_of_layers = None
        self.side = None

    def load(
        self,
        side="left",
        load_transformations=True,
        load_decompositions=True,
        load_cluster=False,
    ):

        print("Loading data ...")

        # config
        path = "results/transformations/" + self.path
        self.side = side
        self.number_of_layers = np.load(path + "number_of_layers.npy").item()

        if load_decompositions:
            # ----  Set path to decompositions
            path = "results/decompositions/" + self.path + side

            for layer in range(self.number_of_layers):

                path_layer = path + "/Layer" + str(layer) + "/"

                # read and write vectors
                self.u_list.append(np.load(path_layer + "u.npy"))
                self.vh_list.append(np.load(path_layer + "vh.npy"))
                self.s_list.append(np.load(path_layer + "s.npy"))
                self.k_list.append(np.load(path_layer + "k.npy").item())

        if load_transformations:
            # ----  Set path to transformations
            path = "results/transformations/" + self.path
            self.labels = np.load(path + "labels.npy")

            # load activations
            for layer in range(self.number_of_layers + 1):

                path_layer = path + "/Layer" + str(layer) + "/"

                # load activations
                self.activation_list.append(np.load(path_layer + "activation.npy"))

                if layer < self.number_of_layers - 1:
                    self.transformation_list.append(
                        np.load(path_layer + "transformation.npy")
                    )

        # ---- Set path to clusters
        if load_cluster:
            path = "results/clusters/" + self.path + side

            # load clusters
            for layer in range(self.number_of_layers):

                path_layer = path + "/Layer" + str(layer) + "/"

                # load activations
                self.clusters.append(
                    (
                        np.load(path_layer + "number_of_clusters.npy"),
                        np.load(path_layer + "clusters.npy"),
                        np.load(path_layer + "center_of_clusters.npy"),
                    )
                )

        return self.side, self.number_of_layers
