from lja.analyser.analyser import Analyser
import numpy as np

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load(side="left", load_features=True)
analyser.plotter.plot_path = "plots/outliers/"
analyser.set_k_per_layer([20, 10, 10, 10])

# Compute computation path on layer level
if True:
    analyser.compute_computation_path_per_layer(True)

if False:
    analyser.compute_computation_path_per_k(True, [352])
    analyser.compute_computation_path_per_k(True, [322])

if False:
    analyser.compute_variation_in_computation()

if True:
    analyser.get_feature_devation(321)
    analyser.get_feature_devation(352)

# good example 713, 718
