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
        self.vector_number = None
        self.plot_path = "plots/" + path
        self.show_plots = show_plots
        self.path = self.plot_path

        # a function that is called as a last step before plotting. It can contain additional augmentations to the origninal plot
        self.custom_function = None

    def set_path(self):
        self.path = self.plot_path

        if self.layer is not None:
            self.path += "Layer" + str(self.layer) + "/"
            if self.vector_number is not None:
                self.path += "Vector" + str(self.vector_number) + "/"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        pass

    def set_layer_and_vector(self, layer, vector_number=None):
        self.layer = layer
        self.vector_number = vector_number
        self.set_path()

        pass

    def present_image(self, title, filename):
        plt.title(title)
        if self.custom_function is not None:
            self.custom_function()

        if filename is not None:
            plt.savefig(self.path + filename + ".png", dpi=300)
        if self.show_plots:
            plt.show()

        pass

    def plot_image(self, image, title="", filename=None, aspect="auto"):

        plt.clf()
        plt.imshow(image, aspect=aspect)
        plt.colorbar()
        plt.clim(-30, 30)
        self.present_image(title, filename)

        pass

    def plot_scatter(self, data, title="", filename=None):

        # plot
        plt.close("all")
        fig, ax = plt.subplots(1)
        sns.scatterplot(
            x="x",
            y="y",
            hue="label",
            style="style",
            palette=sns.color_palette(n_colors=len(set(data["label"]))),
            data=data,
            ax=ax,
            s=40,
        )

        # add labels
        if "text_labels" in list(data.columns):
            for idx, row in data.iterrows():
                plt.text(row["x"], row["y"], row["text_labels"], size=4)

        self.present_image(title, filename)

        pass

    def plot_scatter_simple(
        self, x, y, xlabel="x", ylabel="y", title="", filename=None
    ):

        plt.close("all")
        # plt.plot(x, y, "bo", markersize=2, alpha=1)
        plt.plot(range(int(max(x))), range(int(max(x))), color="red")

        plt.plot(x, y, "bo", markersize=0.1, alpha=0.2)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        self.present_image(title, filename)

        pass

    def plot_reduction(
        self, type, M, labels, text_labels=None, style=None, title="", filename=None
    ):

        # set up path and title
        title = type + ": " + title
        if filename is not None:
            filename = filename + type
        print("\n", title, "\t M-diemnsions: ", M.shape)

        # perfrom reduction
        if type == "tSNE":
            res = TSNE(
                2, init="pca", learning_rate="auto", n_iter=2000, n_jobs=1
            ).fit_transform(M)

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
                "style": "$f$" if style is None else style,
            }
        )

        self.plot_scatter(data, title, filename)

        pass

    def plot_reductions(
        self, M, labels, text_labels=None, style=None, title="", filename="test"
    ):

        # t-SNE plot
        self.plot_reduction("tSNE", M, labels, text_labels, style, title, filename)

        # PCA
        # self.plot_reduction("PCA", M, labels, x, title, filename)

        pass

    def plot_line_plot(self, x, y, title, ylabel, xlabel, filename, xticks=None):

        # plot
        plt.close("all")
        plt.plot(x, y)

        # design
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if xticks is not None:
            plt.xticks(x)

        self.present_image(title, filename)

        pass

    def plot_histogram(self, x, title, filename):

        # plot
        plt.close("all")
        plt.hist(x)

        self.present_image(title, filename)

        pass
