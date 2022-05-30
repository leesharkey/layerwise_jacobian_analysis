import os
import numpy as np
from lja.analyser.plotter import Plotter
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

            elif self.side == "right":
                self.vh_list.append(np.load(os.path.join(path, "vh.npy")))

    def store(self):
        # Set path to decompositions
        path_folder = "results/decompositions/" + self.path + self.side
        print("\nStore in: ", path_folder)

        # Store clusters
        for layer in range(self.number_of_layers):
            newpath = path_folder + "/Layer" + str(layer) + "/"
            np.save(newpath + "number_of_clusters.npy", self.number_of_clusters[layer])
            np.save(newpath + "clusters.npy", self.clusters[layer])

    def get_vectors(self, layer, vector_index):
        if self.side == "left":
            vectors = self.u_list[layer][:, :, vector_index]

        elif self.side == "right":
            vectors = self.vh[:, vector_index, :]

        return vectors

    def cluster_all_vectors(self, n=10):
        for layer in range(self.number_of_layers):
            print("\n\n --- Layer: ", layer)

            clusters = []
            number_of_clusters = []
            for i in range(n):
                print("\nVector:", i)
                n_clusters, cluster_labels = self.cluster_vectors(layer, i)

                clusters.append(cluster_labels)
                number_of_clusters.append(n_clusters)

            self.number_of_clusters.append(number_of_clusters)
            self.clusters.append(cluster_labels)

    def cluster_vectors(self, layer, vector_index, method="spectral"):

        # data
        vectors = self.get_vectors(layer, vector_index)

        # find number of clusters
        n_clusters = self.find_number_of_clusters(layer, vector_index)

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

        # plot
        self.plotter.set_layer(layer)
        self.plotter.plot_reductions(
            vectors,
            cluster_labels,
            None,
            title="U projection_clustered_" + str(vector_index),
            file_name="Vector"
            + str(vector_index)
            + "/U_projections_"
            + str(vector_index)
            + "_"
            + "clustered"
            + "_",
        )

        return n_clusters, cluster_labels

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

        # data for plotting
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
