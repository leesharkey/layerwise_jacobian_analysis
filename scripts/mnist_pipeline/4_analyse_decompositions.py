from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load(side="left")

# general information
analyser.print_shapes()

# decompsotion testing
analyser.create_all_reconstrcution_error_plots()
analyser.create_all_singluarvalue_plots()

k_per_layer = [100, 100, 100, 10]
analyser.test_all_decompositions(k_per_layer)

# plot embeddings
analyser.create_all_reduction_plots(k_per_layer)
