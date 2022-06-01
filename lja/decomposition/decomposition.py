import glob, os
import numpy as np
from sklearn.utils import extmath
import warnings


class Decomposition:
    """Creates an Decomposition object, that calculates singular vectors using regularized, randomized SVD."""

    def __init__(self, path):
        super(Decomposition, self).__init__()

        self.transformations = []
        self.activations = []
        self.decompositions = []
        self.side = None
        self.path = path
        self.load_data()

    def load_data(self):
        path = "results/transformations/" + self.path
        print("Load from: ", path)
        for file in sorted(glob.glob(path + "activation*.npy")):
            self.activations.append(np.load(file))

        for file in sorted(glob.glob(path + "transformation*.npy")):
            self.transformations.append(np.load(file))

    def store(self):
        path = "results/decompositions/" + self.path + self.side + "/"
        print("\nStore in: ", path)
        for i, decomposition in enumerate(self.decompositions):
            newpath = path + "Layer" + str(i) + "/"

            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for item, name in zip(decomposition, ["u", "s", "vh"]):
                np.save(newpath + name + ".npy", item)

    def decompose(self, k_given, side="left"):
        self.decompositions = []
        self.side = side
        print(len(self.transformations))
        for i, T in enumerate(self.transformations):
            if i != len(self.transformations) - 1:
                print("\nLayer:", i)
                u, s, vh, k = self.get_decomposition(T, k_given, self.side)
                self.decompositions.append((u, s, vh))
                self.test_decomposition(
                    u, s, vh, self.activations[i], self.activations[i + 1], k
                )

        return self

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
        tol = 1e-4
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
            acc = np.sum(np.isclose(xp1_hat, xp1, atol=tol)) / xp1_hat.size
            error = np.mean(np.abs(xp1_hat - xp1))

            print(f"Reconstruction accuracy (tol={tol}):", acc)
            print("Reconstruction AbsError:", error)

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
            acc = np.sum(np.isclose(xp1_hat, xp1, atol=tol)) / xp1_hat.size
            error = np.mean(np.abs(xp1_hat - xp1))

            print(f"Reconstruction accuracy (tol={tol}):", acc)
            print("Reconstruction AbsError:", error)

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

            u_stacked, s, vh = extmath.randomized_svd(T_stacked, k, random_state=1)

            # 3. Recover single U Matrices
            U = u_stacked.reshape(T.shape[0], T.shape[1], k)

            return U, s, vh, k

        elif side == "right":
            # 1. stack transformations of each input
            T_stacked = np.hstack(T)

            # 2. Apply SVD
            k = min(k, T.shape[1])
            u, s, vh_stacked = extmath.randomized_svd(T_stacked, k, random_state=1)

            # 3. Recover single VH Matices
            vh_stacked = vh_stacked.reshape(k, T.shape[0], T.shape[2])
            VH = np.hstack(vh_stacked).reshape(T.shape[0], k, T.shape[2])

            return u, s, VH, k

        else:
            raise Exception("Decomposition: invalid side")

    def get_decomposition_by_layer_index(self, index, k, side="left"):
        if index >= len(self.transformations):
            raise Exception("Decomposition: invalid layer index")
        else:
            return self.get_decomposition(self.transformations[index], k, side)
