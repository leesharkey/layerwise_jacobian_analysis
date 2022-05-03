from lja.decomposition.decomposition import Decomposition

decomp = Decomposition("mnist/dropout/")
decomp.decompose(100, "left").store()
# decomp.decompose(10000, "right").store()

# --> right is a bit unprecise
