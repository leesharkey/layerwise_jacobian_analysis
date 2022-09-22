from lja.analyser.analyser import Analyser
import numpy as np

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.plotter.plot_path = "plots/decompositions/"
analyser.load(side="left")

k_per_layer = [20, 10, 10, 10]

# general information
analyser.print_shapes()

# reconstrcution metrics: Creat error plots
if False:
    k_range = range(1, 25)
    analyser.create_all_reconstrcution_error_plots(k_range)
    analyser.create_all_singluarvalue_plots()

# reconstrcution metrics: test specific k per layer
if False:
    analyser.test_all_decompositions(k_per_layer)

# plot embeddings
if True:
    analyser.create_all_reduction_plots(k_per_layer)

# plot all samples of the mnist dataset
if False:
    analyser.plot_all_samples()
