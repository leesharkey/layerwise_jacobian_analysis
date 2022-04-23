from lja.decomposition.decomposition import Decomposition

decomp = Decomposition("logical/xor/")
decomp.decompose(3, "left").store()
decomp.decompose(3, "right").store()
