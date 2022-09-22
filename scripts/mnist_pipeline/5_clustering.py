from lja.clusterer.clusterer import Clusterer

clusterer = Clusterer("mnist/dropout/")
clusterer.load()
clusterer.cluster_all_layers(k=10, n_neighbors=30, number_of_clusters=[13, 10, 10, 10])
clusterer.store()
