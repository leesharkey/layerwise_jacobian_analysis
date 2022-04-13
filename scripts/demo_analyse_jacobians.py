import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import data
x0 = np.load("results/jacobians/logical/input_0.npy")
x1 = np.load("results/jacobians/logical/input_1.npy")
x2 = np.load("results/jacobians/logical/input_2.npy")
x3 = np.load("results/jacobians/logical/input_3.npy")
labels = np.round(x3).flatten()

t0 = np.load("results/jacobians/logical/transformation_0.npy")
t1 = np.load("results/jacobians/logical/transformation_1.npy")
t2 = np.load("results/jacobians/logical/transformation_2.npy")


# 2. Analyse first layer
t = t0
print(x0)
print(labels.astype(int))

# simple descriptive approach
if False:
    c0 = t[labels == 0]
    c1 = t[labels == 1]

    print(np.mean(c0, axis=(1, 2)))
    print(np.mean(c1, axis=(1, 2)))

    print(np.sum(c0, axis=(1, 2)))
    print(np.sum(c1, axis=(1, 2)))


# simple kmeans approach
if False:
    t_flat = t.reshape(16, -1)
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(t_flat)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()

    kmeanModel = KMeans(n_clusters=4).fit(t_flat)
    print(kmeanModel.labels_)


# singular value decomposition
print("\nSVD:")
u, s, vh = np.linalg.svd(t, full_matrices=False)
print(u.shape, s.shape, vh.shape)

if False:
    var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=3)
    var_explained = var_explained[1]

    sns.barplot(
        x=list(range(1, len(var_explained) + 1)), y=var_explained, color="limegreen"
    )
    plt.xlabel("SVs", fontsize=16)
    plt.ylabel("Percent Variance Explained", fontsize=16)
    plt.savefig("svd_scree_plot.png", dpi=100)


# right singular vector
vh_major = vh[:, 0, :]

if False:
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(vh_major)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()

kmeanModel = KMeans(n_clusters=2).fit(vh_major)
print(kmeanModel.labels_)


# left singular vector
u_major = u[:, :, 0]

if True:
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(u_major)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()

kmeanModel = KMeans(n_clusters=2).fit(u_major)
print(kmeanModel.labels_)
