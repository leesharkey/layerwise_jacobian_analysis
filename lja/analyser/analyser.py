import glob, os
import numpy as np
from lja.analyser.plotter import Plotter
from lja.analyser.dataloader import Dataloader
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
        self.data = Dataloader(path)
        self.number_of_layers = None
        self.side = None

    def load(self, side="left", load_cluster=False):

        # load transformations and decompositions
        self.side, self.number_of_layers = self.data.load(
            side, load_cluster=load_cluster
        )

        pass

    def reduce_activations(self, layer):

        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            self.data.activation_list[layer],
            self.data.labels,
            title="Input projections",
            filename="activation_",
        )

        pass

    def reduce_write_matrix(self, layer, k):

        # data
        U = self.data.u_list[layer][:, :, :k]
        U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            U_flatten,
            self.data.labels,
            title="U projections",
            filename="U_projections_",
        )

        pass

    def reduce_write_vector(self, layer, vector_index):

        # data
        U = self.data.u_list[layer][:, :, vector_index]

        self.plotter.set_layer_and_vector(layer, vector_index)
        self.plotter.plot_reductions(
            U,
            self.data.labels,
            title="U projection_" + str(vector_index),
            filename="U_projections_" + str(vector_index) + "_",
        )

        pass

    def create_all_reduction_plots(self, k_per_layer=None, n=10):

        if k_per_layer is None:
            k_per_layer = self.data.k_list

        for layer in range(self.number_of_layers):
            print("\nLayer:", layer)
            self.reduce_write_matrix(layer, k_per_layer[layer])
            self.reduce_activations(layer)

            for vector_index in range(n):
                self.reduce_write_vector(layer, vector_index)

        pass

    def print_shapes(self):

        print("\nSide:", self.side)
        for layer in range(self.number_of_layers):

            print("\nLayer:", layer)
            print("Input:", self.data.activation_list[layer].shape)
            print("Read Vectors:", self.data.vh_list[layer].shape)
            print("Write Vectors:", self.data.u_list[layer].shape)

        pass

    # ---- Test the decompositions ----

    def create_all_singluarvalue_plots(self):

        for layer in range(self.number_of_layers):
            self.create_singluarvalue_plot(layer)

        pass

    def create_singluarvalue_plot(self, layer):

        singular_values = self.data.s_list[layer]

        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_line_plot(
            range(len(singular_values)),
            singular_values,
            "Singular Values",
            "Value",
            "k",
            "singular_values",
        )
        pass

    def test_all_decompositions(self, k_per_layer=None):

        for layer in range(self.number_of_layers):

            print("\nLayer:", layer)

            if k_per_layer is None:
                k = self.data.k_list[layer]
            else:
                k = k_per_layer[layer]

            self.test_decomposition(layer, k)

    def test_decomposition(self, layer, k):

        # declare
        u = self.data.u_list[layer]
        s = self.data.s_list[layer]
        vh = self.data.vh_list[layer]
        x = self.data.activation_list[layer]
        xp1 = self.data.activation_list[layer + 1]

        # truncuate
        if self.side == "left":
            u = u[:, :, :k]
            vh = vh[:k, :]
            s = s[:k]

        # elif self.side == "right":
        # todo

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
            # todo
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

    def create_all_reconstrcution_error_plots(self):

        for layer in range(self.number_of_layers):

            k_range = np.linspace(1, self.data.k_list[layer], 10).astype(int)
            self.create_reconstrcution_error_plot(layer, k_range)

    def create_reconstrcution_error_plot(self, layer, k_range):

        # memory
        errors = []
        accuracies = []

        for k in k_range:

            print("\n-- K:", k)

            # obtain decomposition
            acc, error = self.test_decomposition(layer, k)

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
            xticks=k_range,
        )
        self.plotter.plot_line_plot(
            k_range,
            accuracies,
            "Reconstruction Accuracy",
            "Accuracy",
            "Number of components",
            "reconstruction_accuracy",
            xticks=k_range,
        )

        pass

    # ---- Test the clustering ----

    def create_cluster_plot(self, layer):

        (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer]

        self.plotter.plot_image(cluster_labels, title="", filename=None, aspect="auto")

        pass

    def print_cluster_shapes(self):
        print("\nSide:", self.side)
        for layer in range(self.number_of_layers):

            #  cluster infos
            (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer]
            sizes = []

            for cluster in range(cluster_n):

                # size of cluster
                mask = cluster_labels == cluster
                sizes.append(np.sum(mask))

            print("\nLayer:", layer)
            print("Number of clusters:", cluster_n)
            print("Number of clusters: ", sizes)

        pass
