import glob, os
import numpy as np
from sklearn.utils import extmath


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
        print("Store in: ", path)
        for i, decomposition in enumerate(self.decompositions):
            newpath = path + "Layer" + str(i) + "/"

            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for item, name in zip(decomposition, ["u", "s", "vh"]):
                np.save(newpath + name + ".npy", item)

    def decompose(self, k, side="left"):
        self.decompositions = []
        self.side = side
        for T in self.transformations:
            u, s, vh = self.get_decomposition(T, k, self.side)
            self.decompositions.append((u, s, vh))

    def get_decomposition(self, T, k, side):

        if side == "left":
            # 1. stack tranformations of each input
            T_stacked = np.vstack(T)

            # 2. Apply SVD
            k = min(k, T.shape[2])
            u_stacked, s, vh = extmath.randomized_svd(T_stacked, k, random_state=1)
            print(u_stacked.shape, s.shape, vh.shape)

            # 3. Recover single U Matices
            U = u_stacked.reshape(T.shape[0], T.shape[1], k)

            return U, s, vh

        elif side == "right":
            # 1. stack tranformations of each input
            T_stacked = np.hstack(T)
            print(T_stacked.shape)

            # 2. Apply SVD
            k = min(k, T.shape[1])
            u, s, vh_stacked = extmath.randomized_svd(T_stacked, k, random_state=1)
            print(u.shape, s.shape, vh_stacked.shape)

            # 3. Recover single VH Matices
            VH = vh_stacked.reshape(T.shape[0], k, T.shape[2])

            return u, s, VH

        else:
            raise Exception("Decomposition: invalid side")

    def get_decomposition_by_layer_index(self, index, k, side="left"):
        if index >= len(self.transformations):
            raise Exception("Decomposition: invalid layer index")
        else:
            return self.get_decomposition(self.transformations[index], k, side)
