import os
import numpy as np
from lja.analyser.plotter import Plotter
from lja.analyser.dataloader import Dataloader
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from numpy.linalg import norm


class Constructor:
    """Creates an feature visualisation object, that visualises the read vectors of the decompositions."""

    def __init__(self, path, target, show_plots=False):
        super(Constructor, self).__init__()

        self.path = path
        self.plotter = Plotter("/features/", show_plots)
        self.data = Dataloader(path)
        self.number_of_layers = None
        self.side = None
        self.target = target
        self.feature_memory = []
        self.feature_memory_available = []

    def load(self, side="left"):

        # load transformations and decompositions
        self.side, self.number_of_layers = self.data.load(side, load_cluster=True)

        self.u_list = self.data.u_list
        self.vh_list = self.data.vh_list
        self.k_list = self.data.k_list
        self.s_list = self.data.s_list

        pass

    def set_k_per_layer(self, k_per_layer):

        self.k_list = k_per_layer
        self.u_list = []
        self.vh_list = []

        for layer in range(self.number_of_layers):
            k = self.k_list[layer]
            u = self.data.u_list[layer][:, :, :k]
            vh = self.data.vh_list[layer][:k, :]

            if layer == 0:
                input_dimension = vh.shape[1] - 1

            feature_memory = np.zeros((u.shape[0], k, input_dimension))
            feature_memory_available = np.zeros((u.shape[0], k))

            self.u_list.append(u)
            self.vh_list.append(vh)
            self.feature_memory.append(feature_memory)
            self.feature_memory_available.append(feature_memory_available)

    def set_plot_path2(self, layer, feature_index, target_index):
        self.plotter.set_layer_and_vector(layer, feature_index)
        self.plot_path = "by_" + str(self.target) + "/granularity_" + self.granularity
        newpath = self.plotter.path + "/" + self.plot_path
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        pass

    def set_plot_path(self, layer, feature_index, target_index):
        self.plotter.set_layer_and_vector(layer, target_index)
        self.plot_path = ""
        newpath = self.plotter.path
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        pass

    def set_results_path(self, layer, feature_index, create_folder=True):
        self.results_path = (
            "results/features/"
            + self.path
            + self.side
            + "/Layer"
            + str(layer)
            + "/Vector"
            + str(feature_index)
            + "/by_"
            + str(self.target)
            + "/granularity_"
            + self.granularity
        )
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        pass

    def get_filename(self, feature_index, target_index, mode):
        name = (
            "/feature_"
            + str(feature_index)
            + "_"
            + str(self.target)
            + "_"
            + str(target_index)
        )

        if mode == "plot":
            filename = self.plot_path + name

        elif mode == "file":
            filename = self.results_path + name + ".npy"

        else:
            filename = ""

        return filename

    def load_feature(self, layer, feature_index, target_index):
        """
        Loads an already computed feature from the disk
        """
        # set path
        self.set_results_path(layer, feature_index, create_folder=False)
        file_path = self.get_filename(feature_index, target_index, "file")

        # load if possible
        if os.path.exists(file_path):
            feature = np.load(file_path, allow_pickle=True)
            return feature
        else:
            return None

    def store_feature(self, feature, layer, feature_index, target_index):
        """
        Stores a feature to the disk as numpy array
        """
        # set path
        self.set_results_path(layer, feature_index)
        np.save(
            self.get_filename(feature_index, target_index, "file"), feature,
        )

        pass

    def plot_feature(self, feature, layer, feature_index, target_index):
        """
        PLots the feature as heat-map
        """
        # set path
        self.set_plot_path(layer, feature_index, target_index)
        filename = self.get_filename(feature_index, target_index, "plot")
        file_path = self.plotter.path + filename + ".png"

        if not os.path.exists(file_path):
            self.plotter.plot_image(
                feature.reshape(28, 28),
                title="Layer: "
                + str(layer)
                + " Feature: "
                + str(feature_index)
                + " "
                + str(self.target)
                + ": "
                + str(target_index),
                filename=filename,
            )

        pass

    def construct_single_feature(
        self,
        layer,
        feature_index,
        target_index,
        plot=True,
        store=True,
        reuse_stored_features=True,
        store_all_computed_features=True,
    ):
        """
        Constructs a single feature.
        layer                           - the layer of the read vector to be visualised
        feature_index,                  - the index of the read vector to be visualised
        target_index,                   - the index of the sample/profile/profile_cluster those write vectors be used for reconstruction
        plot=True,                      - whether a heat-map of the featuer should be stores
        store=True,                     - whether the feature should be stored
        reuse_stored_features           - whether if already computed features should be loaded. Faster if enabled. Should be disabled for testing/dev.
        store_all_computed_features     - whether the features that are computed on the fly should be stored (slows down the computation)
        """

        # 0. Check if already computed:
        feature = None
        already_stored = False

        if reuse_stored_features:

            # test if already loaded and in memory
            if self.feature_memory_available[layer][target_index, feature_index] == 1:
                feature = self.feature_memory[layer][target_index, feature_index, :]
                already_stored = True

            if reuse_stored_features:
                feature = self.load_feature(layer, feature_index, target_index)

                if feature is not None:
                    self.feature_memory_available[layer][
                        target_index, feature_index
                    ] = 1
                    self.feature_memory[layer][target_index, feature_index] = feature
                    already_stored = True

        # Compute feature if not pre-computed
        if feature is None:

            # 1. Select the read vector
            read_vector = self.vh_list[layer][feature_index, :-1]

            if layer == 0:

                # 2. Finished
                feature = read_vector

            else:

                # 2. Get the write vectors that we use to approximate the read vector
                write_vector_candidates = self.get_write_vector_candidates(
                    layer, target_index
                )

                # 4. Compute similarity between the write_vectors and the read_vector
                similarity = (
                    np.dot(write_vector_candidates, read_vector)
                    / (norm(write_vector_candidates, axis=1) * norm(read_vector))
                    # * self.s_list[layer][feature_index]
                )

                # 5. Recursive idea: pick features of the previous layer as constructors

                # 5.1. Choose the corresponding target_index for the pervious layer
                target_index_next = self.get_corresponding_target_index(
                    layer, target_index
                )

                # 5.2. Collect constrcutor vectors
                construction_vectors = []
                number_write_vectors = len(write_vector_candidates)
                for write_vector_index in range(number_write_vectors):
                    construction_vectors.append(
                        self.construct_single_feature(
                            layer - 1,
                            write_vector_index,
                            target_index_next,
                            plot=False,
                            store=store_all_computed_features,
                            reuse_stored_features=reuse_stored_features,
                        )
                    )
                construction_vectors = np.array(construction_vectors)

                # 6. Construct the feature as linear combination the previous features
                construction_vectors_weighted = (
                    np.diag(similarity) @ construction_vectors
                )
                # Assumption add corresponding singular value for scalig
                construction_vectors_combined = np.sum(
                    construction_vectors_weighted, axis=0
                )
                feature = construction_vectors_combined

        if plot:
            if True:
                feature_masked = feature * self.data.activation_list[0][target_index]
                self.plot_feature(feature_masked, layer, feature_index, target_index)
            else:
                self.plot_feature(feature, layer, feature_index, target_index)

        if store and not already_stored:
            self.store_feature(feature, layer, feature_index, target_index)

        return feature

    def construct_multiple_features(
        self,
        layers,
        feature_indices,
        target_indices,
        granularites=["profile_cluster"],
        plot=True,
        store=True,
        reuse_stored_features=True,
    ):
        """
        Construct multiple features based on the list contents.
        granularites  - defines which U matrix should be used for reconstruction one of ['sample', 'profile', 'profile_cluster']

        See above
        """

        config_memory = ["-"]
        for (granularity, layer, feature_index, target_index) in itertools.product(
            granularites, layers, feature_indices, target_indices
        ):

            # verbose
            config = [granularity, layer, feature_index]
            if config_memory != config:
                print("\nGranularity: ", granularity)
                print("Layer: ", layer)
                print("Feature: ", feature_index)

                # update granularity:
                if config_memory[0] != config[0]:
                    self.set_granularity(granularity)

                # remember
                config_memory = config

            # construction
            feature = self.construct_single_feature(
                layer,
                feature_index,
                target_index,
                plot=plot,
                store=store,
                reuse_stored_features=reuse_stored_features,
            )
        pass


class ConstructorBySample(Constructor):
    def __init__(self, path, granularity="sample", show_plots=False):
        Constructor.__init__(self, path, "sample", show_plots)
        self.set_granularity(granularity)

    def set_granularity(self, granularity):
        if granularity in [
            "sample",
            "profile",
            "layer_average",
            "vector_average",
            "sample_average",
        ]:
            self.granularity = granularity
        else:
            raise Exception(
                "Invaild granularity argument \n it must be one of [sample, profile"
            )

        pass

    def get_write_vector_candidates(self, layer, sample_index):
        """
        returns a set of write vectors that is used to match the read vector
        """

        if self.granularity == "sample":

            # 2. Pick in the u-matrix of the sample directly
            write_vector_candidates = self.u_list[layer - 1][sample_index].T

        elif self.granularity == "profile":

            # 2. Pick the u-vector center the sample belongs to for each dimension seperately

            # pick the profile the sample belongs to
            (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer - 1]
            profile = cluster_labels[sample_index, :]

            # pick the profle centers in the U vector space of the profile
            write_vector_candidates = cluster_centers[profile]

        return write_vector_candidates

    def get_corresponding_target_index(self, layer, sample_index):
        """
        returns the profile_index of the next reconstruction call.
        This function defines which visualisations are used to construct the subsequent features
        (from hidden to output direction)
        """

        # the corresponding feature is ined by the sample index again: easy mapping
        return sample_index


class ConstructorByProfile(Constructor):
    def __init__(self, path, granularity="profile", show_plots=False):
        Constructor.__init__(self, path, "profile", show_plots)
        self.set_granularity(granularity)

    def set_granularity(self, granularity):

        if granularity in ["profile"]:
            self.granularity = granularity
        else:
            raise Exception(
                "Invaild granularity argument \n it must be one of [profile]"
            )

        pass

    def get_write_vector_candidates(self, layer, profile_index):
        """
        returns a set of write vectors that is used to match the read vector
        """

        if self.granularity == "profile":

            # pick profile
            (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer - 1]
            unique_profiles = np.unique(cluster_labels, axis=0)
            profile = unique_profiles[profile_index, :]

            # pick the profle centers in the U vector space of the profile
            write_vector_candidates = cluster_centers[profile]

        return write_vector_candidates

    def get_corresponding_target_index(self, layer, profile_index):

        """
        returns the profile_index of the next reconstruction call.
        This function defines which visualisations are used to construct the subsequent features
        (from hidden to output direction)
        """

        if layer == 1:

            # end of recursion expected
            target_index_next = profile_index

        else:

            # 1. Find samples of the profile that needs to be mapped
            # 1.2 pick profile to be mapped
            (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer - 1]

            unique_profiles = np.unique(cluster_labels, axis=0)
            profile = unique_profiles[profile_index, :]

            # 1.3 identify samples of that profile; this is a mask
            members_profile = np.equal(cluster_labels, profile).all(axis=1)

            # 2. Find the profiles of the samples in the previous layer
            (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer - 2]
            previous_profiles = cluster_labels[members_profile, :]

            # 3. Pick one of these profiles as the next target, by the method of majority vote:
            # TODO: better methods?
            values, counts = np.unique(previous_profiles, axis=0, return_counts=True)
            most_frequent_previous_profile = values[np.argmax(counts)]

            # 4. Find the index of this profile to be passed on
            unique_profiles = np.unique(cluster_labels, axis=0)
            (target_index_next,) = np.where(
                np.equal(unique_profiles, most_frequent_previous_profile).all(axis=1)
            )
            target_index_next = target_index_next.item()

        return target_index_next
