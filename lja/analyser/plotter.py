from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings


class Plotter:
    """Creates an plotting object, that can create different plots for a matrixc or a list of matrices."""

    def __init__(self):
        super(Plotter, self).__init__()
        warnings.simplefilter(action="ignore", category=FutureWarning)

    def plot_results(self, data, title):

        # plot
        fig, ax = plt.subplots(1)
        sns.scatterplot(x="x", y="y", hue="label", data=data, ax=ax, s=120)
        ax.set_aspect("equal")
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        # add labels
        for idx, row in data.iterrows():
            plt.text(row["x"], row["y"], row["label_string"])

        plt.title(title)
        plt.show()

    def plot_reduction(self, type, M, labels, x, title):

        title = type + ": " + title
        print("\n", title, "\t M-diemnsions: ", M.shape)

        # convert input data to strings for labels
        x = np.round(x, 2)
        x_string = [",".join(str(item2) for item2 in item) for item in x]

        # perfrom reduction
        if type == "tSNE":
            res = TSNE(2, init="pca", learning_rate="auto").fit_transform(M)

        elif type == "PCA":
            res = PCA(n_components=2).fit_transform(M)

        else:
            raise Exception("Plotter: reduction type not implemented")

        data = pd.DataFrame(
            {
                "x": res[:, 0],
                "y": res[:, 1],
                "label": labels[:, 0],
                "label_string": x_string,
            }
        )

        self.plot_results(data, title)

    def plot_reductions(self, M, labels, x, title):

        # t-SNE plot
        self.plot_reduction("tSNE", M, labels, x, title)

        # PCA
        self.plot_reduction("PCA", M, labels, x, title)
