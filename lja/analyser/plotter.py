from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os


class Plotter:
    """Creates an plotting object, that can create different plots for a matrixc or a list of matrices."""

    def __init__(self, path, show_plots=False):
        super(Plotter, self).__init__()
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.layer = None
        self.plot_path = "plots/" + path
        self.show_plots = show_plots
        self.path = self.plot_path

        # a function that is called as a last step before plotting. It can contain additional augmentations to the origninal plot
        self.custom_function = None

    def set_layer(self, layer):
        self.path = self.plot_path + "Layer" + str(layer) + "/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if layer != self.layer:
            print("Save plots in:", self.path)
        self.layer = layer

    def present_image(self, title, file_name):
        plt.title(title)
        if self.custom_function is not None:
            self.custom_function()

        if file_name is not None:
            plt.savefig(self.path + file_name + ".png", dpi=500)
        if self.show_plots:
            plt.show()

    def plot_image(self, image, title="", file_name=None, aspect="auto"):

        plt.clf()
        plt.imshow(image, aspect=aspect)
        plt.colorbar()
        self.present_image(title, file_name)

    def plot_scatter(self, data, title, file_name=None):

        # plot
        plt.close("all")
        fig, ax = plt.subplots(1)
        sns.scatterplot(
            x="x",
            y="y",
            hue="label",
            palette=sns.color_palette(n_colors=len(set(data["label"]))),
            data=data,
            ax=ax,
            s=20,
        )

        # add labels
        if "text_labels" in list(data.columns):
            for idx, row in data.iterrows():
                plt.text(row["x"], row["y"], row["text_labels"])

        self.present_image(title, file_name)

    def plot_reduction(self, type, M, labels, text_labels, title, file_name=None):

        # set up path and title
        title = type + ": " + title
        file_name = file_name + type
        print("\n", title, "\t M-diemnsions: ", M.shape)

        # perfrom reduction
        if type == "tSNE":
            res = TSNE(2, init="pca", learning_rate="auto", n_iter=2000).fit_transform(
                M
            )

        elif type == "PCA":
            res = PCA(n_components=2).fit_transform(M)

        else:
            raise Exception("Plotter: reduction type not implemented")

        # store results in a dataframe
        data = pd.DataFrame(
            {
                "x": res[:, 0],
                "y": res[:, 1],
                "label": labels,
                "text_labels": np.repeat("", len(labels))
                if text_labels is None
                else text_labels,
                "style": "$f$",
            }
        )

        self.plot_scatter(data, title, file_name)

    def plot_reductions(self, M, labels, x, title, file_name=None):

        # t-SNE plot
        self.plot_reduction("tSNE", M, labels, x, title, file_name)

        # PCA
        # self.plot_reduction("PCA", M, labels, x, title, file_name)
