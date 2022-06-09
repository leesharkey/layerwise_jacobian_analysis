import os
import numpy as np
from lja.analyser.plotter import Plotter
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import SpectralClustering
import scipy


class ProfileClusterer:
    """Creates an clustering object, that clusters the read vector CLUSTERS of each layer."""

    def __init__(self, path, show_plots=False):
        super(ProfileClusterer, self).__init__()
        self.path = path
        self.plotter = Plotter(path, show_plots)
        self.number_of_layers = None
        self.side = None

        self.u_list = []

        self.number_of_clusters = []
        self.clusters = []
        self.center_of_clusters = []

        self.profile_clusters = []
        self.number_of_profile_clusters = []
        self.center_of_profile_clusters = []

    def load_data(self, side="left"):

        # config
        self.side = side

        # Set path to decompositions
        path_folder = "results/decompositions/" + self.path + self.side
        print("Load decompositions from: ", path_folder)

        # Load clusters
        self.number_of_layers = len(next(os.walk(path_folder))[1])
        for layer in range(self.number_of_layers):
            path = path_folder + "/Layer" + str(layer) + "/"

            # clusters of write vectors
            self.clusters.append(
                np.load(os.path.join(path, "clusters.npy"), allow_pickle=True)
            )
            self.number_of_clusters.append(
                np.load(os.path.join(path, "number_of_clusters.npy"))
            )
            self.center_of_clusters.append(
                np.load(os.path.join(path, "center_of_clusters.npy"), allow_pickle=True)
            )

            # original write vectors
            self.u_list.append(np.load(os.path.join(path, "u.npy")))

        pass

    def store(self):

        # Set path to decompositions
        path_folder = "results/decompositions/" + self.path + self.side
        print("\nStore in: ", path_folder)

        # Store clusters
        for layer in range(self.number_of_layers):
            newpath = path_folder + "/Layer" + str(layer) + "/"

            # (1)
            np.save(
                newpath + "number_of_profile_clusters.npy",
                self.number_of_profile_clusters[layer],
            )

            # (k, n)
            np.save(newpath + "profile_clusters.npy", self.profile_clusters[layer])

            # (k, n, dim)
            np.save(
                newpath + "center_of_profile_clusters.npy",
                np.array(self.center_of_profile_clusters[layer], dtype=object),
            )

        pass

    def cluster_all_profiles(self, plot=False):
        for layer in range(self.number_of_layers):

            print("\n\n --- Layer: ", layer)

            # 1. Find number of clusters
            affintiy_matrix, n_clusters = self.find_number_of_clusters(layer, plot=plot)
            self.number_of_profile_clusters.append(n_clusters)
            print("Number of clusters:", n_clusters)

            # 2. Clustering
            cluster_labels = self.cluster_profiles(layer, n_clusters, affintiy_matrix)
            self.profile_clusters.append(cluster_labels)

            # 3. Compute centers
            centers = self.get_center_of_clusters(layer, cluster_labels, n_clusters)
            self.center_of_profile_clusters.append(centers)

            # 4. Plot
            if plot:
                self.plot_cluster_embedding(layer, centers, cluster_labels)

        pass

    def plot_cluster_embedding(self, layer, centers, cluster_labels):

        # 1 Original data
        U = self.u_list[layer]
        U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])
        cluster_text_labels = np.repeat("", len(cluster_labels))

        # 2 Centers
        centers = np.swapaxes(centers, 1, 2)
        centers_flatten = centers.reshape(
            centers.shape[0], centers.shape[1] * centers.shape[2]
        )
        center_text_labels = [str(item) for item in range(len(centers))]
        center_labels = range(len(centers))

        # 3. Combine data
        data = np.append(U_flatten, centers_flatten, axis=0)
        cluster_labels = np.append(cluster_labels, center_labels)
        text_labels = np.append(cluster_text_labels, center_text_labels)

        # 4. Plot
        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            data,
            cluster_labels,
            text_labels=text_labels,
            title="U projections_clustered",
            file_name="U_projections_clustered_",
        )

    def get_center_of_clusters(self, layer, cluster_labels, n_clusters):

        # select profiles
        profile_centers = self.center_of_clusters[layer]  # (n_fimensions, n_clusters)
        profiles = self.clusters[layer]
        profiles_cluster_centers = []

        # loop through clusters
        for label in range(n_clusters):

            # we choose the profiles that belong to that cluster
            mask = cluster_labels == label
            profile_cluster = profiles[:, mask]

            # cluster centers is a list of centers of each dimension (n_dimensions)
            cluster_center = []

            # loop thorugh u-vectors/profile dimensions
            for profiles_in_dimension_i, profile_centers_in_dimension_i in zip(
                profile_cluster, profile_centers,
            ):

                # convert to numpy
                profile_centers_in_dimension_i = np.array(
                    profile_centers_in_dimension_i
                )

                # select the centers of the profiles in dimension i
                centers_in_dimension_i = profile_centers_in_dimension_i[
                    profiles_in_dimension_i
                ]

                # compute the center of these centers in dimension i #meta-clustering
                mean_center_in_dimension_i = np.mean(centers_in_dimension_i, axis=0)
                cluster_center.append(mean_center_in_dimension_i)

            profiles_cluster_centers.append(cluster_center)

        return profiles_cluster_centers  # (number_of_clusters, k, dimension_of_layer)

    def find_number_of_clusters(
        self,
        layer,
        max_number_of_clusters=20,
        size_of_candidate_clusters=1,
        plot=False,
    ):

        # select profiles
        profiles = self.clusters[layer]

        # create similarity matrix based on shared profiles
        number_samples = profiles.shape[1]
        affintiy_matrix = np.zeros((number_samples, number_samples))
        for s1 in range(number_samples):
            for s2 in range(s1, number_samples):
                similarity = np.sum(profiles[:, s1] == profiles[:, s2])
                affintiy_matrix[s1, s2] = similarity
                affintiy_matrix[s2, s1] = similarity

        # eigengap heursitic
        A = affintiy_matrix
        L = scipy.sparse.csgraph.laplacian(A, normed=True)
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
            self.plotter.set_layer_and_vector(layer)
            self.plotter.plot_scatter(
                data,
                title="eigengap plot: " + str(gaps_indices),
                file_name="eigengap_",
            )

        return affintiy_matrix, optimal_gap

    def cluster_profiles(self, layer, n_clusters, affintiy_matrix=None):

        # select profiles
        profiles = self.clusters[layer]

        # create similarity matrix based on shared profiles
        if affintiy_matrix is None:
            number_samples = profiles.shape[1]
            affintiy_matrix = np.zeros((number_samples, number_samples))
            for s1 in range(number_samples):
                for s2 in range(s1, number_samples):
                    similarity = np.sum(profiles[:, s1] == profiles[:, s2])
                    affintiy_matrix[s1, s2] = similarity
                    affintiy_matrix[s2, s1] = similarity

        # clustering
        clusterer = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", n_neighbors=3
        )
        cluster_labels = clusterer.fit_predict(affintiy_matrix)

        return cluster_labels
