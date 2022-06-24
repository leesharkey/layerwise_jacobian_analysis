from lja.decomposition.decomposition import Decomposition

# Create Decomposition Object
decomp = Decomposition("mnist/dropout/")
decomp.load()

# decompose
decomp.decompose([600, 600, 600, 10], "left")
decomp.store()
