from lja.decomposition.decomposition import Decomposition

# Create Decomposition Object
decomp = Decomposition("mnist/dropout/")
decomp.load()

# Create the error plots
if False:
    decomp.get_error_function(
        0, [10, 100, 200, 300, 400, 500, 600, 700, 800], side="left"
    )

    decomp.get_error_function(
        1, [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], side="left"
    )

    decomp.get_error_function(
        2, [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], side="left"
    )

    decomp.get_error_function(3, [1, 2, 4, 6, 8, 10], side="left")

# decompose
decomp.decompose([600, 600, 600, 10], "left")
decomp.store()
