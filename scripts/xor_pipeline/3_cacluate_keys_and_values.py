from lja.decomposition.decomposition import Decomposition

decomp = Decomposition("logical/xor/")
decomp.decompose(100, "right")
decomp.store()
