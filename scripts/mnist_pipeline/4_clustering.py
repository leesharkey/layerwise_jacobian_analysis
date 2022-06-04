from lja.clusterer.clusterer import Clusterer

clusterer = Clusterer("mnist/dropout/")
clusterer.load_data()
clusterer.cluster_all_vectors()
clusterer.store()
