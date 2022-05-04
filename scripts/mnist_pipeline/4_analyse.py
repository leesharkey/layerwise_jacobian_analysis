from lja.analyser.analyser import Analyser
import numpy as np
import matplotlib.pyplot as plt

analyser = Analyser("mnist/dropout/", show_plots=True)

analyser.load_data(side="left", layer=0)
# analyser.analyse_decomposition()

# analyser.activation_plots()
# analyser.read_in_scaled_plots()
# analyser.stacked_matrix_plots()
# analyser.visualize_read_vectors()

analyser.print_shapes()
analyser.visualize_write_vector()


"""
analyser.print_shapes()
f_index = 9

feature = analyser.vh[f_index, :]
feature_scaled = feature * analyser.s[f_index]
print(feature_scaled.shape)

analyser.plotter.plot_image(feature_scaled[0:-1].reshape(28, 28), "test")

# for sample 1
samples = [10, 110, 210, 310, 410, 510, 610, 710, 810, 910]
U = analyser.u[:, :, f_index]
U[U < -0.004] = -1
U[U > 0.004] = 1

# the write vector is:
analyser.plotter.plot_image(U, "test")
"""
