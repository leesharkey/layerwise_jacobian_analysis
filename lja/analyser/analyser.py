import glob, os
import numpy as np
from lja.analyser.plotter import Plotter
import scipy.stats as stats
import math
from scipy.spatial import distance_matrix
import pandas as pd


class Analyser:
    """Creates an analysis object, that can provide different statistics about the decompositions."""

    def __init__(self, path, show_plots=False):
        super(Analyser, self).__init__()
        self.path = path
        self.plotter = Plotter(path, show_plots)

        self.side = None
        self.number_of_layers = None

        # sample level
        self.u_list = []
        self.vh_list = []
        self.labels = None
        self.activation_list = []

        # profile level
        self.number_of_clusters = []
        self.clusters = []

        # profile cluster level
        self.number_of_profile_clusters = []
        self.profile_clusters = []

    def load_data(self, side="left"):

        # config
        self.side = side

        # Set path to decompositions
        path_folder = "results/decompositions/" + self.path + self.side
        print("Load decompositions from: ", path_folder)

        # Load decompsitions
        self.number_of_layers = len(next(os.walk(path_folder))[1])
        for layer in range(self.number_of_layers):
            path = path_folder + "/Layer" + str(layer) + "/"

            # read and write vectors
            self.u_list.append(np.load(os.path.join(path, "u.npy")))
            self.vh_list.append(np.load(os.path.join(path, "vh.npy")))

            # clusters of write vectors
            self.clusters.append(
                np.load(os.path.join(path, "clusters.npy"), allow_pickle=True)
            )
            self.number_of_clusters.append(
                np.load(os.path.join(path, "number_of_clusters.npy"))
            )

            # clusters of profiles
            self.profile_clusters.append(
                np.load(os.path.join(path, "profile_clusters.npy"), allow_pickle=True)
            )
            self.number_of_profile_clusters.append(
                np.load(
                    os.path.join(path, "number_of_profile_clusters.npy"),
                    allow_pickle=True,
                )
            )

        # load class labels
        path = "results/transformations/" + self.path
        self.labels = np.load(os.path.join(path, "labels.npy"))

        # load activations
        for layer in range(self.number_of_layers):
            self.activation_list.append(
                np.load(os.path.join(path, "activation_" + str(layer) + ".npy"))
            )

        pass

    def reduce_write_matrix(self, layer):

        # data
        U = self.u_list[layer]
        U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            U_flatten, self.labels, title="U projections", file_name="U_projections_",
        )

        pass

    def reduce_activations(self, layer):

        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            self.activation_list[layer],
            self.labels,
            title="Input projections",
            file_name="activation_",
        )

        pass

    def reduce_write_vector(self, layer, vector_index):

        # data
        U = self.u_list[layer][:, :, vector_index]

        self.plotter.set_layer_and_vector(layer, vector_index)
        self.plotter.plot_reductions(
            U,
            self.labels,
            title="U projection_" + str(vector_index),
            file_name="U_projections_" + str(vector_index) + "_",
        )

        pass

    def create_all_plots(self, n=10):

        for layer in range(self.number_of_layers):
            print("\nLayer:", layer)
            self.reduce_write_matrix(layer)
            self.reduce_activations(layer)

            for vector_index in range(n):
                self.reduce_write_vector(layer, vector_index)

        pass

    def print_shapes(self):

        print("\nSide:", self.side)
        for layer in range(self.number_of_layers):

            print("\nLayer:", layer)
            print("Input:", self.activation_list[layer].shape)
            print("Read Vectors:", self.vh_list[layer].shape)
            print("Write Vectors:", self.u_list[layer].shape)

        pass

    def print_profile_cluster_infos(self):

        for layer in range(self.number_of_layers):

            #  general infos
            n_cluster = self.number_of_profile_clusters[layer]
            cluster_labels = self.profile_clusters[layer]

            print("\n\n---Layer:", layer)
            print("Number of layers: ", n_cluster)

            # cluster infos
            for cluster in range(n_cluster):

                # size of cluster
                mask = cluster_labels == cluster
                size = np.sum(mask)

                # samples in that cluster
                labels_of_cluster_samples = self.labels[mask]
                values, counts = np.unique(
                    labels_of_cluster_samples, return_counts=True
                )
                most_frequent = values[np.argsort(counts)]

                print("\nCluster:", cluster)
                print("Size:", size)
                print("Most Frequent labels:\t", most_frequent)
                print("Frequencies:\t\t", np.sort(counts))

        pass
