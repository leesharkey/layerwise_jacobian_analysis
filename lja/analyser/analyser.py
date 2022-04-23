import glob, os
import numpy as np
from lja.analyser.plotter import Plotter
import scipy.stats as stats
import math
from scipy.spatial import distance_matrix


class Analyser:
    """Creates an analysis object, that can provide different statistics about the decompositions."""

    def __init__(self, path):
        super(Analyser, self).__init__()
        self.path = path
        self.side = None
        self.s = None
        self.u = None
        self.vh = None
        self.activation = None
        self.activation_p1 = None
        self.labels = None

    def load_data(self, side, layer):
        # config
        self.side = side
        self.layer = layer

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

        # Load activation
        path = "results/transformations/" + self.path
        print("Load from: ", path)

        self.activation = np.load(
            os.path.join(path, "activation_" + str(layer) + ".npy")
        )
        self.activation_p1 = np.load(
            os.path.join(path, "activation_" + str(layer + 1) + ".npy")
        )

        # Load labels
        labels = np.load(
            os.path.join(
                path,
                "activation_" + str(len(glob.glob(path + "activation_*")) - 1) + ".npy",
            )
        )
        self.labels = np.round(labels)

    def analyse_decomposition(self):
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

            np.set_printoptions(suppress=True)
            print("\nAnalysis with left-stacked SV:")
            print("The read vectors are (row-wise): \n", np.round(vh, 3))
            print(
                "\nThe read in vectors are \n(i-th row describes the representation in the i-th dimension, \nj-th column describes the representation of the j-th input):\n",
                np.round(read_in, 3),
            )
            print("\nThe scaled read in vectors are:\n", np.round(read_in_scaled, 3))
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
            print("\nThe scaled read in vectors are:\n", np.round(read_in_scaled, 3))
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

    def stacked_matrix_plots(self):
        plotter = Plotter()

        if self.side == "left":
            U = self.u
            U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

            plotter.plot_reductions(
                U_flatten, self.labels, self.activation, "U projections"
            )

        elif self.side == "right":
            VH = self.vh
            VH_flatten = VH.reshape(VH.shape[0], VH.shape[1] * VH.shape[2])

            plotter.plot_reductions(
                VH_flatten, self.labels, self.activation, "VH projections"
            )

    def activation_plots(self):
        plotter = Plotter()

        plotter.plot_reductions(
            self.activation, self.labels, self.activation, "Input projections"
        )

    def read_in_scaled_plots(self):
        plotter = Plotter()

        plotter.plot_reductions(
            self.read_in_scaled.T,
            self.labels,
            self.activation,
            "Scaled read in projections",
        )
