from lja.analyser.analyser import Analyser

analyser = Analyser("logical/xor/")

analyser.load_data("right", 1)
analyser.analyse_decomposition()

analyser.activation_plots()
analyser.read_in_scaled_plots()
analyser.stacked_matrix_plots()
