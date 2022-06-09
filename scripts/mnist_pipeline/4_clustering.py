from lja.clusterer.clusterer import Clusterer
from lja.clusterer.profile_clusterer import ProfileClusterer

clusterer = Clusterer("mnist/dropout/")
clusterer.load_data()
clusterer.cluster_all_vectors()
clusterer.store()

profile_clusterer = ProfileClusterer("mnist/dropout/")
profile_clusterer.load_data()
profile_clusterer.cluster_all_profiles(plot=True)
profile_clusterer.store()
