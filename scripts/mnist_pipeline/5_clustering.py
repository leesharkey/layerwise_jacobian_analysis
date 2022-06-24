from lja.clusterer.clusterer import Clusterer

clusterer = Clusterer("mnist/dropout/")
clusterer.load()
clusterer.cluster_all_layers(k=5, n_neighbors=10, number_of_clusters=[7, 7, 9, 5])
clusterer.store()

# last layer fails
# clusterer.cluster_one_layer(3, k=5, plot=True, n_neighbors=10, number_of_clusters=5)
