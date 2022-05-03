from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)

analyser.load_data(side="left", layer=0)
# analyser.analyse_decomposition()

analyser.activation_plots()
analyser.read_in_scaled_plots()
analyser.stacked_matrix_plots()
analyser.visualize_read_vectors()

# todo: commits
# todo: do experiment with varying n, k
