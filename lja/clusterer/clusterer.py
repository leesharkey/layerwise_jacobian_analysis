import os
import numpy as np
from lja.analyser.plotter import Plotter
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import SpectralClustering
import scipy


class Clusterer:
    """Creates an clustering object, that clusters the read vectors of each layer."""

    def __init__(self, path, show_plots=False):
        super(Clusterer, self).__init__()
        self.path = path
        self.side = None
        self.u_list = []
        self.vh_list = []
        self.clusters = []
        self.number_of_clusters = []
        self.number_of_layers = None
        self.plotter = Plotter(path, show_plots)
        self.ks = []
        self.center_of_clusters = []

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

            if self.side == "left":
                self.u_list.append(np.load(os.path.join(path, "u.npy")))
                self.ks.append(self.u_list[0].shape[2])

            elif self.side == "right":
                self.vh_list.append(np.load(os.path.join(path, "vh.npy")))
                self.ks.append(self.vh_list[0].shape[2])

        pass

    def store(self):
        # Set path to decompositions
        path_folder = "results/decompositions/" + self.path + self.side
        print("\nStore in: ", path_folder)

        # layers, vectors, num_clusters, 1024
        # Store clusters
        for layer in range(self.number_of_layers):
            newpath = path_folder + "/Layer" + str(layer) + "/"

            # (1)
            np.save(newpath + "number_of_clusters.npy", self.number_of_clusters[layer])

            # (k, n)
            np.save(newpath + "clusters.npy", self.clusters[layer])

            # (k, n, dim)
            np.save(
                newpath + "center_of_clusters.npy",
                np.array(self.center_of_clusters[layer], dtype=object),
            )

        pass

    def get_vectors(self, layer, vector_index):
        if self.side == "left":
            vectors = self.u_list[layer][:, :, vector_index]

        elif self.side == "right":
            vectors = self.vh[:, vector_index, :]

        return vectors

    def cluster_all_vectors(self, n=0):
        for layer in range(self.number_of_layers):

            print("\n\n --- Layer: ", layer)
            clusters = []
            number_of_clusters = []
            centers_list = []
            k = self.ks[layer]

            for vector_index in range(k):
                print("\nVector:", vector_index)
                plot = vector_index < n

                # 1. Find number of clusters
                n_clusters = self.find_number_of_clusters(
                    layer, vector_index, plot=plot
                )
                number_of_clusters.append(n_clusters)

                # 2. Clustering
                cluster_labels = self.cluster_vectors(layer, vector_index, n_clusters)
                clusters.append(cluster_labels)

                # 3. Compute centers
                centers = self.get_center_of_clusters(
                    layer, vector_index, cluster_labels, n_clusters
                )
                centers_list.append(centers)

                # 4. Plot
                if plot:
                    self.plot_cluster_embedding(
                        layer, vector_index, cluster_labels, centers
                    )

            # Store
            self.number_of_clusters.append(number_of_clusters)
            self.clusters.append(clusters)
            self.center_of_clusters.append(centers_list)

        pass

    def get_center_of_clusters(self, layer, vector_index, cluster_labels, n_clusters):

        # data
        vectors = self.get_vectors(layer, vector_index)
        centers = []

        # loop thorugh clusters
        for label in range(n_clusters):
            mask = cluster_labels == label
            cluster = vectors[mask, :]
            center = np.mean(
                cluster, axis=0
            )  #  mean center, maybe try trimmed mean center
            centers.append(center)

        return centers

    def plot_cluster_embedding(self, layer, vector_index, cluster_labels, centers):

        # data
        vectors = self.get_vectors(layer, vector_index)

        # add center labels
        text_labels = [str(item) for item in range(len(centers))]
        text_labels = np.append(np.repeat("", len(cluster_labels)), text_labels)

        # center corrdinates
        vectors = np.append(vectors, centers, axis=0)
        cluster_labels = np.append(cluster_labels, range(len(centers)))

        # plot clusters
        self.plotter.set_layer(layer)
        self.plotter.plot_reductions(
            vectors,
            cluster_labels,
            text_labels,
            title="U projection_clustered_" + str(vector_index),
            file_name="Vector"
            + str(vector_index)
            + "/U_projections_"
            + str(vector_index)
            + "_"
            + "clustered"
            + "_",
        )

        pass

    def cluster_vectors(self, layer, vector_index, n_clusters, method="spectral"):

        # data
        vectors = self.get_vectors(layer, vector_index)

        # clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(vectors)

        elif method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=10
            )
            cluster_labels = clusterer.fit_predict(vectors)

        # stats
        silhouette_avg = silhouette_score(vectors, cluster_labels)

        print(
            "Number of Clusters:",
            n_clusters,
            " - The average silhouette_score is :",
            silhouette_avg,
        )

        return cluster_labels

    def find_number_of_clusters(
        self,
        layer,
        vector_index,
        max_number_of_clusters=50,
        size_of_candidate_clusters=3,
        plot=False,
    ):

        # data
        vectors = self.get_vectors(layer, vector_index)

        # clustering
        clusterer = SpectralClustering(
            n_clusters=2, affinity="nearest_neighbors", n_neighbors=10
        )
        cluster_labels = clusterer.fit_predict(vectors)

        # eigengap heursitic
        A = clusterer.affinity_matrix_
        L = scipy.sparse.csgraph.laplacian(A, normed=True).todense()
        eigenvalues, _ = np.linalg.eig(L)
        eigenvalues = eigenvalues[0:max_number_of_clusters]

        # compute gaps without considering only one cluster
        gaps = np.diff(eigenvalues[1:])

        # order gaps and correct indices
        gaps_indices = np.argsort(gaps)[::-1] + 2

        # select top x candidates and order from small to large
        gaps_indices = np.sort(gaps_indices[:size_of_candidate_clusters])
        optimal_gap = gaps_indices[0]

        # data for plotting the first n
        if plot:
            data = pd.DataFrame(
                {
                    "x": np.arange(len(eigenvalues)) + 1,
                    "y": eigenvalues,
                    "label": np.ones(len(eigenvalues)),
                    "style": "$f$",
                }
            )

            self.plotter.set_layer(layer)
            self.plotter.plot_scatter(
                data,
                title="eigengap plot: " + str(gaps_indices),
                file_name="Vector"
                + str(vector_index)
                + "/eigengap_"
                + str(vector_index)
                + "_",
            )

        return optimal_gap
