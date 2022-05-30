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
        self.side = None
        self.layer = None
        self.s = None
        self.u = None
        self.vh = None
        self.k = None
        self.activation = None
        self.activation_p1 = None
        self.labels = None
        self.plotter = Plotter(path, show_plots)

    def load_data(self, side, layer, k="all"):
        # config
        self.side = side
        self.layer = layer
        self.k = k
        self.plotter.set_layer(layer)

        # Load decompositions
        path = (
            "results/decompositions/"
            + self.path
            + self.side
            + "/Layer"
            + str(layer)
            + "/"
        )
        print("Load from: ", path)

        self.s = np.load(os.path.join(path, "s.npy"))
        self.u = np.load(os.path.join(path, "u.npy"))
        self.vh = np.load(os.path.join(path, "vh.npy"))
        self.cluster_labels = np.load(os.path.join(path, "clusters.npy"))

        # TODO: take subet of vectors

        # Load activation
        path = "results/transformations/" + self.path
        print("Load from: ", path)

        self.activation = np.load(
            os.path.join(path, "activation_" + str(layer) + ".npy")
        )
        self.activation_p1 = np.load(
            os.path.join(path, "activation_" + str(layer + 1) + ".npy")
        )
        self.labels = np.load(os.path.join(path, "labels.npy"))

    def analyse_decomposition(self, verbose=True):
        u = self.u
        vh = self.vh
        s = self.s
        x = self.activation
        xp1 = self.activation_p1

        if self.side == "left":
            U = u
            U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

            # 1. What is read:
            x_ext = np.insert(x, x.shape[1], 1, axis=1)
            read_in = vh @ x_ext.T

            # 2. Scale
            read_in_scaled = np.diag(s) @ read_in
            self.read_in_scaled = read_in_scaled

            # 3. Write to ouput
            t = U @ read_in_scaled
            xp1_hat = t[range(t.shape[0]), :, range(t.shape[0])]
            acc = np.sum(np.isclose(xp1_hat, xp1, atol=1e-04)) / xp1_hat.size

            if verbose:
                np.set_printoptions(suppress=True)
                print("\nAnalysis with left-stacked SV:")
                print("The read vectors are (row-wise): \n", np.round(vh, 3))
                print(
                    "\nThe read in vectors are \n(i-th row describes the representation in the i-th dimension, \nj-th column describes the representation of the j-th input):\n",
                    np.round(read_in, 3),
                )
                print(
                    "\nThe scaled read in vectors are:\n", np.round(read_in_scaled, 3)
                )
                print(
                    "\nThe reconstructed output is:\n(i-th row describes activation for i-th input\nj-th column descibribes the activation of j-th neuron)\n",
                    np.round(xp1_hat, 3),
                )

                print("\nReconstruction accuracy is: ", acc)

                print(
                    "\nThe write vectors depend on the inputs. \n(the i-th row describes the output to the i-th neuron \nthe j-th column describes the j-th write vector)\n",
                    np.round(U, 3),
                )

                print(
                    "\nThe euclidean distance matrix for the U matrices is:\n",
                    np.round(np.triu(distance_matrix(U_flatten, U_flatten)), 3),
                )
                print()

        elif self.side == "right":
            VH = vh
            VH_flatten = VH.reshape(VH.shape[0], VH.shape[1] * VH.shape[2])

            # 1. What is read:
            x_ext = np.insert(x, x.shape[1], 1, axis=1)
            read_ins = VH @ x_ext.T

            # pick first column of first read_in and so on
            read_in = np.transpose(
                read_ins[range(read_ins.shape[0]), :, range(read_ins.shape[0])]
            )

            # 2. Scale
            read_in_scaled = np.diag(s) @ read_in
            self.read_in_scaled = read_in_scaled

            # 3. Write to ouput
            xp1_hat = np.transpose(u @ read_in_scaled)
            acc = np.sum(np.isclose(xp1_hat, xp1, atol=1e-04)) / xp1_hat.size

            if verbose:
                np.set_printoptions(suppress=True)
                print("\nAnalysis with right-stacked SV:")
                print(
                    "The read vectors depend on the inputs. The read vectors are (row-wise): \n",
                    np.round(VH, 3),
                )
                print(
                    "\nThe read in vectors are \n(i-th row describes the representation in the i-th dimension, \nj-th column describes the representation of the j-th input):\n",
                    np.round(read_in, 3),
                )
                print(
                    "\nThe scaled read in vectors are:\n", np.round(read_in_scaled, 3)
                )
                print(
                    "\nThe reconstructed output is:\n(i-th row describes activation for i-th input\nj-th column descibribes the activation of j-th neuron)\n",
                    np.round(xp1_hat, 3),
                )
                print("\nReconstruction accuracy is: ", acc)
                print(
                    "\nThe write vectors are: \n(the i-th row describes the output to the i-th neuron \nthe j-th column describes the j-th write vector)\n",
                    np.round(u, 3),
                )

                print(
                    "\nThe euclidean distance matrix for the VH matrices is:\n",
                    np.round(np.triu(distance_matrix(VH_flatten, VH_flatten)), 3),
                )
                print()

    def reduce_write_matrix(self):
        U = self.u
        U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

        self.plotter.plot_reductions(
            U_flatten,
            self.labels,
            self.activation,
            title="U projections",
            file_name="U_projections_",
        )

    def activation_plots(self):
        self.plotter.plot_reductions(
            self.activation,
            self.labels,
            self.activation,
            title="Input projections",
            file_name="activation_",
        )

    def read_in_scaled_plots(self):
        self.analyse_decomposition(verbose=False)
        self.plotter.plot_reductions(
            self.read_in_scaled.T,
            self.labels,
            self.activation,
            title="Scaled read in projections",
            file_name="read_in_",
        )

    def visualize_read_vector(self, vector_index, threshold=None, scaled=True):
        if self.layer == 0:
            vector = self.vh[vector_index, :-1]
            vector = vector.reshape(28, 28)
        else:
            vector = self.vh[vector_index, :]
            vector = np.expand_dims(vector, axis=0)

        if threshold is not None:
            vector = self.apply_threshold(vector, threshold)
        elif scaled:
            vector = vector * self.s[vector_index]

        self.plotter.plot_image(
            vector,
            title="Read vector" + str(vector_index),
            file_name="Vector"
            + str(vector_index)
            + "/read_vector"
            + str(vector_index)
            + "_"
            + str(threshold)
            + "_",
        )

    def visualize_write_vector(self, vector_index, threshold=None):
        U = self.u[:, :, vector_index]

        if threshold is not None:
            U = self.apply_threshold(U, threshold)

        self.plotter.plot_image(
            U,
            title="Write vector" + str(vector_index),
            file_name="Vector"
            + str(vector_index)
            + "/write_vector"
            + str(vector_index)
            + "_"
            + str(threshold)
            + "_",
        )

    def reduce_write_vector(self, vector_index, labels=None, annotation="classes"):
        U = self.u[:, :, vector_index]

        self.plotter.plot_reductions(
            U,
            self.labels if (labels is None) else labels,
            self.activation,
            title="U projection_" + str(vector_index),
            file_name="Vector"
            + str(vector_index)
            + "/U_projections_"
            + str(vector_index)
            + "_"
            + str(annotation)
            + "_",
        )

    def visualize_vectors(self, n=10, read_threshold=None, write_threshold=None):
        for i in range(n):

            # create directories
            path = self.plotter.path + "Vector" + str(i)
            if not os.path.exists(path):
                os.makedirs(path)

            # call submethods
            self.visualize_write_vector(i)
            self.visualize_read_vector(i)

            if write_threshold is not None:
                self.visualize_write_vector(i, write_threshold)

            if read_threshold is not None:
                self.visualize_read_vector(i, read_threshold)

    def reduce_all_write_vectors(self, n=10):
        for i in range(n):

            # t-SNE
            self.reduce_write_vector(i)

    def print_shapes(self):

        print("\nSide:", self.side)
        print("Input:", self.activation.shape)
        print("Output:", self.activation_p1.shape)

        print("\nVH:", self.vh.shape)
        print("s:", self.s.shape)
        print("U:", self.u.shape)

    def apply_threshold(self, X, quantile):
        pos_q = np.quantile(X[X > 0], 1 - quantile)
        X[X > pos_q] = 1

        neg_q = np.quantile(X[X < 0], quantile)
        X[X < neg_q] = -1

        X[np.abs(X) != 1] = 0

        return X

    def create_all_plots(self, n=10, read_threshold=None, write_threshold=None):
        self.activation_plots()
        self.read_in_scaled_plots()
        self.reduce_write_matrix()
        self.visualize_vectors(
            n=n, write_threshold=write_threshold, read_threshold=read_threshold
        )
        self.reduce_all_write_vectors()
