import glob, os
import numpy as np
from lja.analyser.plotter import Plotter
from lja.analyser.dataloader import Dataloader
import scipy.stats as stats
import math
from scipy.spatial import distance_matrix
import pandas as pd
from sklearn.metrics import pairwise_distances
from numpy.linalg import norm


class Analyser:
    """Creates an analysis object, that can provide different statistics about the decompositions."""

    def __init__(self, path, show_plots=False):
        super(Analyser, self).__init__()

        self.path = path
        self.plotter = Plotter(path, show_plots)
        self.data = Dataloader(path)
        self.number_of_layers = None
        self.side = None

    def load(self, side="left", load_cluster=False, load_features=False):

        # load transformations and decompositions
        self.side, self.number_of_layers = self.data.load(
            side, load_cluster=load_cluster, load_features=load_features
        )

        pass

    # -- Plotting MNIST datset
    def plot_all_samples(self):

        # custom path!
        self.plotter.path = "plots/mnist_dataset/"

        for i, sample in enumerate(self.data.activation_list[0]):

            sample = sample.reshape(28, 28)
            self.plotter.plot_image(
                sample,
                title="Sample " + str(i),
                filename="sample_" + str(i),
                aspect="auto",
            )

        self.plotter.set_path()

        pass

    def plot_single_sample(self, i):

        sample = self.data.activation_list[0][i]
        sample = sample.reshape(28, 28)
        self.plotter.plot_image(
            sample,
            title="Sample " + str(i),
            filename="sample_" + str(i),
            aspect="auto",
        )

        pass

    # -- Util
    def get_misclassificatin_text_labels(self, samples=range(1000)):

        # add labels for misclassification:
        samples = np.array(samples)
        text_labels = np.repeat("", len(samples)).tolist()
        for index in samples[self.data.misclassification_mask]:
            string = str(self.data.predictions[index]) + "-" + str(index)
            text_labels[index] = string

        return text_labels

    def print_shapes(self):

        print("\nSide:", self.side)
        for layer in range(self.number_of_layers):

            print("\nLayer:", layer)
            print("Input:", self.data.activation_list[layer].shape)
            print("Read Vectors:", self.data.vh_list[layer].shape)
            print("Write Vectors:", self.data.u_list[layer].shape)

        pass

    def set_k_per_layer(self, k_per_layer):
        self.data.k_list = k_per_layer

    # -- Reduction Methods
    def reduce_write_matrix(self, layer, k):

        # data
        U = self.data.u_list[layer][:, :, :k]
        U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

        # add misclassification labels
        text_labels = self.get_misclassificatin_text_labels()

        # plot
        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            U_flatten,
            self.data.labels,
            title="U projections",
            filename="U_projections_",
            style=self.data.misclassification_mask == False,
            text_labels=text_labels,
        )

        pass

    def reduce_write_vector(self, layer, vector_index):

        # data
        U = self.data.u_list[layer][:, :, vector_index]

        # add misclassification labels
        text_labels = self.get_misclassificatin_text_labels()

        # plot
        self.plotter.set_layer_and_vector(layer, vector_index)
        self.plotter.plot_reductions(
            U,
            self.data.labels,
            title="U projection_" + str(vector_index),
            filename="U_projections_" + str(vector_index) + "_",
            style=self.data.misclassification_mask == False,
            text_labels=text_labels,
        )

        pass

    def reduce_activations(self, layer):

        # add misclassification labels
        text_labels = self.get_misclassificatin_text_labels()

        # plot
        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            self.data.activation_list[layer],
            self.data.labels,
            title="Input projections",
            filename="activation_",
            style=self.data.misclassification_mask == False,
            text_labels=text_labels,
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

    # ---- Test decompositions ----

    def create_all_singluarvalue_plots(self):

        for layer in range(self.number_of_layers):
            self.create_singluarvalue_plot(layer)

        pass

    def create_singluarvalue_plot(self, layer, k=50):

        singular_values = self.data.s_list[layer][:k]

        singular_values = singular_values / np.sum(singular_values)

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

        return self.compare_activation_and_prediction(
            layer, xp1, xp1_hat, mode="reconstruction", plot=True
        )

    def create_all_reconstrcution_error_plots(self, k_range):

        for layer in range(self.number_of_layers):
            self.create_reconstrcution_error_plot(layer, k_range)

    def create_reconstrcution_error_plot(self, layer, k_range):

        # memory
        errors = []
        accuracies = []
        correlations = []
        dot_products = []

        for k in k_range:

            print("\n-- K:", k)

            # obtain decomposition
            error, acc, cor, dot = self.test_decomposition(layer, k)

            # store
            errors.append(error)
            accuracies.append(acc)
            correlations.append(cor)
            dot_products.append(dot)

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
            correlations,
            "Reconstruction Correlation",
            "Correlation",
            "Number of components",
            "reconstruction_correlation",
            xticks=k_range,
        )
        pass

    def scale(self, A):
        return (A - np.min(A)) / (np.max(A) - np.min(A))

    def compare_activation_and_prediction(
        self, layer, target, pred, mode="activation", plot=True
    ):

        # scale to interval 0-1
        pred = pred.flatten()
        target = target.flatten()

        # error metrics
        error = np.mean(np.abs(target - pred))
        acc = np.mean(np.isclose(target, pred, atol=0.1))
        cor = np.corrcoef(target, pred)[0][1]
        dot = np.dot(target, pred)

        # print info
        print("\nLayer:", layer)
        print("Error:", format(error, ".2f"))
        print("Correlation:", format(cor, ".3f"))
        print("Dot product:", format(dot, ".0f"))

        # plot
        if plot:
            self.plotter.set_layer_and_vector(layer)

            if mode == "activation":
                self.plotter.plot_scatter_simple(
                    target.flatten(),
                    pred.flatten(),
                    "activation",
                    "predicted activation",
                    "Predicted vs. Actual Activation",
                    "activation_correlation",
                )

            else:
                self.plotter.plot_scatter_simple(
                    target.flatten(),
                    pred.flatten(),
                    "activation",
                    "reconstructed activation",
                    "Reconstrcutedvs. Actual Activation",
                    "reconstrcution_correlation",
                )

        return error, acc, cor, dot

    # ---- Features  ----
    def reduce_feature_matrix(self, layer):

        # data
        features = self.data.feature_list[layer]

        # reshape
        features = features.reshape(-1, features.shape[1] * features.shape[2])

        # add misclassification labels
        text_labels = self.get_misclassificatin_text_labels()

        # plot
        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            features,
            self.data.labels,
            title="Features",
            filename="feature_",
            style=self.data.misclassification_mask == False,
            text_labels=text_labels,
        )

        pass

    def reduce_feature(self, layer, feature_index):

        # data
        features = self.data.feature_list[layer][:, feature_index]

        # add misclassification labels
        text_labels = self.get_misclassificatin_text_labels()

        # plot
        self.plotter.set_layer_and_vector(layer, feature_index)
        self.plotter.plot_reductions(
            features,
            self.data.labels,
            title="Features " + str(feature_index),
            filename="feature_" + str(feature_index),
            style=self.data.misclassification_mask == False,
            text_labels=text_labels,
        )

        pass

    def create_all_feature_reduction_plots(self, n=10):

        for layer in range(1, self.number_of_layers):
            print("\nLayer:", layer)
            self.reduce_feature_matrix(layer)

            for feature_index in range(n):
                self.reduce_feature(layer, feature_index)

        pass

    def get_class_centers_per_feature_dimension(self, layer):

        n_classes = len(np.unique(self.data.labels))
        k = self.data.k_list[layer]

        # centers
        u_centers = np.zeros((k, n_classes, self.data.u_list[layer].shape[1]))
        feature_centers = np.zeros(
            (k, n_classes, self.data.feature_list[layer].shape[2])
        )

        # loop through feature dimensions
        for feature_index in range(k):

            # loop through classes
            for c in range(n_classes):
                class_range = np.where(self.data.labels == c)[0]

                # select U matrices and compute mean
                U = self.data.u_list[layer][class_range, :, feature_index]
                u_centers[feature_index, c] = np.mean(U, axis=0)

                # select features matrices and compute mean
                features = self.data.feature_list[layer][class_range, feature_index, :]
                feature_centers[feature_index, c] = np.mean(features, axis=0)

        return u_centers, feature_centers

    def get_class_centers_per_layer(self, layer):

        n_classes = len(np.unique(self.data.labels))
        k = self.data.k_list[layer]

        # centers
        u_centers_list = []
        feature_centers_list = []

        # loop through classes
        for c in range(n_classes):
            class_range = np.where(self.data.labels == c)[0]

            # select U matrices and compute mean
            U = self.data.u_list[layer][class_range, :, :k]
            U = U.reshape(-1, U.shape[1] * U.shape[2])  # flatten
            u_centers_list.append(np.mean(U, axis=0))

            # select features matrices and compute mean
            features = self.data.feature_list[layer][class_range, :k, :]
            features = features.reshape(
                -1, features.shape[1] * features.shape[2]
            )  # flatten
            feature_centers_list.append(np.mean(features, axis=0))

        return u_centers_list, feature_centers_list

    def create_feature_average_plot(self, layer):

        (u_centers, feature_centers) = self.get_class_centers_per_feature_dimension(
            layer
        )

        for feature_index in range(self.data.k_list[layer]):

            # loop over clases:
            for c in np.unique(self.data.labels):

                # compute mean feature
                mean_feature = feature_centers[feature_index, c].reshape(28, 28)
                self.plotter.set_layer_and_vector(layer, feature_index)
                self.plotter.plot_image(
                    mean_feature,
                    title="Feature for class " + str(c),
                    filename="feature_class_" + str(c),
                    aspect="auto",
                )

                if layer == 0:
                    break

        pass

    def create_all_feature_average_plots(self):

        for layer in range(self.number_of_layers):
            print("\nLayer:", layer)
            self.create_feature_average_plot(layer)

        pass

    def test_features(self, baseline=False):

        # input data
        inputs = self.data.activation_list[0]

        # store to return
        targets = []
        preds = []

        # loop through  layers
        for layer in range(self.number_of_layers):

            k = self.data.k_list[layer]

            # extract decompositions
            target = self.data.activation_list[layer + 1]
            pred = np.zeros_like(target)
            singular_values = self.data.s_list[layer]
            write_vectors = self.data.u_list[layer]

            similarities = []

            # loop through samples and features

            for sample_index in range(len(inputs)):

                for feature_index in range(k):

                    feature = self.data.feature_list[layer][sample_index, feature_index]

                    # similarity
                    if baseline:
                        similarity = 1
                    else:
                        similarity = np.dot(feature, inputs[sample_index])

                        # if layer in [1, 2]:
                        # similarity *= 0.68

                    similarities.append(similarity)

                    additional_scale = 1
                    # if layer == 1:
                    #    additional_scale = 1.35
                    # if layer == 2:
                    #    additional_scale = 1.51

                    # prediction
                    pred[sample_index] += (
                        similarity
                        * write_vectors[sample_index, :, feature_index]
                        * singular_values[feature_index]
                        * additional_scale
                    )

            targets.append(target)
            preds.append(pred)

            self.compare_activation_and_prediction(layer, target, pred)
            print("Mean Similarity:", format(np.mean(similarities), ".2f"))

        return targets, preds

    def test_features_with_validation(
        self, inputs, activations, labels, baseline=False
    ):

        # loop through  layers
        for layer in range(self.number_of_layers):

            k = self.data.k_list[layer]

            # extract decompositions
            target = activations[layer + 1]
            pred = np.zeros_like(target)
            singular_value = self.data.s_list[layer]
            write_vectors = self.data.u_list[layer][:, :, :k]
            features = self.data.feature_list[layer]

            # loop through samples and features
            for sample_index in range(len(inputs)):
                class_range = range(
                    labels[sample_index] * 100, labels[sample_index] * 100 + 100
                )

                for feature_index in range(k):

                    # compute mean of class
                    mean_feature = np.mean(
                        features[class_range, feature_index, :], axis=0
                    )
                    mean_write_vector = np.mean(
                        write_vectors[class_range, :, feature_index], axis=0
                    )

                    # similarity
                    if baseline:
                        similarity = 1
                    else:
                        similarity = np.dot(mean_feature, inputs[sample_index])

                    # predictions
                    pred[sample_index] += (
                        similarity * mean_write_vector * singular_value[feature_index]
                    )

            self.compare_activation_and_prediction(layer, target, pred)

        pass

    def get_feature_devation(self, sample_index):

        # obtain actual label
        label = self.data.labels[sample_index]
        mask = self.data.activation_list[0][sample_index] > 0

        for layer in range(1, self.number_of_layers):

            # obtain mean features
            (_, mean_feature) = self.get_class_centers_per_feature_dimension(layer)

            differences = []

            # loop through feature dimensions
            for feature_index in range(self.data.k_list[layer]):
                feature = self.data.feature_list[layer][sample_index, feature_index]

                # compare
                difference = np.abs(feature - mean_feature[feature_index, label])
                differences.append(difference)

                # plot
                difference_masked = difference * mask
                self.plotter.set_layer_and_vector(layer, feature_index)
                self.plotter.plot_image(
                    difference_masked.reshape(28, 28),
                    title="Difference for sample " + str(sample_index),
                    filename="difference_sample_" + str(sample_index),
                    aspect="auto",
                )

            # plot averaged difference
            difference = np.mean(differences, axis=0)
            difference_masked = difference * mask
            self.plotter.set_layer_and_vector(layer)
            self.plotter.plot_image(
                difference_masked.reshape(28, 28),
                title="Difference for sample " + str(sample_index),
                filename="difference_sample_" + str(sample_index),
                aspect="auto",
            )

        pass

    # ---- Analyse computations ----
    def compute_closest_center(self, candidate, centers):
        D = pairwise_distances(candidate, centers)[0]
        return np.argmin(D)

    def compute_computation_path_per_layer(self, printing=False, samples=None):

        # 1. Compute class centers per feature
        u_centers_list = []
        feature_centers_list = []

        for layer in range(self.number_of_layers):
            u_centers, feature_centers = self.get_class_centers_per_layer(layer)
            u_centers_list.append(u_centers)
            feature_centers_list.append(feature_centers)

        # memories
        n_samples = len(self.data.labels)
        u_computation_path = np.zeros((n_samples, self.number_of_layers))
        feature_computation_path = np.zeros((n_samples, self.number_of_layers))

        if samples is None:
            samples = range(n_samples)

        for sample_index in samples:

            for layer in range(self.number_of_layers):

                # params
                k = self.data.k_list[layer]

                # 2. Compute nearest center on all sub dimensions:
                U = self.data.u_list[layer][sample_index, :, :k]
                U = U.reshape(1, U.shape[0] * U.shape[1])
                u_computation_path[sample_index, layer] = self.compute_closest_center(
                    U, u_centers_list[layer]
                )

                feature = self.data.feature_list[layer][sample_index, :k, :]
                feature = feature.reshape(1, feature.shape[0] * feature.shape[1])
                feature_computation_path[
                    sample_index, layer
                ] = self.compute_closest_center(feature, feature_centers_list[layer])

            if printing:
                print("\nSample:", sample_index)
                print("Read in:  ", feature_computation_path[sample_index])
                print("Write out:", u_computation_path[sample_index])

        return u_computation_path, feature_computation_path

    # analyse when final decision is made
    def get_layer_of_final_decision(self, feature_side=False):

        # obtain computation path
        (
            u_computation_path,
            feature_computation_path,
        ) = self.compute_computation_path_per_layer()

        # get final decisions
        output = u_computation_path[:, self.number_of_layers - 1]

        # memory
        final_decision_layer = np.repeat(5, (len(output)))
        mask = output == output

        # define layers to consider
        if feature_side:
            layer_range = range(1, self.number_of_layers)
        else:
            layer_range = range(0, self.number_of_layers)

        # compute first findign of that decision without break in betweeen
        for layer in layer_range[::-1]:

            # decison of that layer
            if feature_side:
                decision = feature_computation_path[:, layer]
            else:
                decision = u_computation_path[:, layer]

            # correct decsion and all previous decision correct
            mask = (decision == output) & mask
            final_decision_layer[mask] = layer

        return final_decision_layer

    def compute_computation_path_per_k(self, printing=False, samples=None):

        # 1. Compute class centers per feature
        u_centers_list = []
        feature_centers_list = []

        for layer in range(self.number_of_layers):
            (u_centers, feature_centers) = self.get_class_centers_per_feature_dimension(
                layer
            )
            u_centers_list.append(u_centers)
            feature_centers_list.append(feature_centers)

        # memory
        n_samples = len(self.data.labels)
        k = 10
        u_computation_path = np.zeros((n_samples, self.number_of_layers, k))
        feature_computation_path = np.zeros((n_samples, self.number_of_layers, k))

        # lop trough samples
        if samples is None:
            samples = range(n_samples)

        for sample_index in samples:

            if printing:
                print("\nSample:", sample_index)

            for layer in range(self.number_of_layers):

                # loop through feature dimensions
                for feature_index in range(k):

                    # Nearest u center
                    U = self.data.u_list[layer][sample_index, :, feature_index].reshape(
                        1, -1
                    )
                    centers = u_centers_list[layer][feature_index]
                    u_computation_path[
                        sample_index, layer, feature_index
                    ] = self.compute_closest_center(U, centers)

                    # Nearest feature center
                    feature = self.data.feature_list[layer][
                        sample_index, feature_index, :
                    ].reshape(1, -1)
                    centers = feature_centers_list[layer][feature_index]
                    feature_computation_path[
                        sample_index, layer, feature_index
                    ] = self.compute_closest_center(feature, centers)

                if printing:
                    print("\nLayer:", layer)
                    if layer > 0:
                        print(
                            "Read in:   ", feature_computation_path[sample_index, layer]
                        )
                    print("Write out: ", u_computation_path[sample_index, layer])

        return u_computation_path, feature_computation_path

    def compute_variation_in_computation(self, feature_side=False):

        # obtain computation path
        (
            u_computation_path,
            feature_computation_path,
        ) = self.compute_computation_path_per_k()

        # get final decisions
        output = u_computation_path[:, self.number_of_layers - 1, 0]
        n_samples = len(self.data.labels)

        # define layers to consider
        if feature_side:
            layer_range = range(1, self.number_of_layers)
        else:
            layer_range = range(0, self.number_of_layers)

        # memory
        difference_to_output = np.zeros((len(output), len(layer_range)))

        for i, layer in enumerate(layer_range):
            for sample_index in range(n_samples):

                if feature_side:
                    vector = feature_computation_path[sample_index, layer]
                else:
                    vector = u_computation_path[sample_index, layer]

                # compute difference between vector and output
                difference_to_output[sample_index, i] = np.sum(
                    vector != output[sample_index]
                )

        return difference_to_output

    # ---- Test clustering ----

    def create_cluster_plot(self, layer):

        (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer]

        self.plotter.plot_image(cluster_labels, title="", filename=None, aspect="auto")

        pass

    def reduce_write_matrix_with_cluster_labels(self, layer, k):

        (cluster_n, cluster_labels, cluster_centers) = self.data.clusters[layer]

        # data
        U = self.data.u_list[layer][:, :, :k]
        U_flatten = U.reshape(U.shape[0], U.shape[1] * U.shape[2])

        self.plotter.set_layer_and_vector(layer)
        self.plotter.plot_reductions(
            U_flatten,
            cluster_labels,
            title="U projections",
            filename="U_projections_clustered",
        )

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
