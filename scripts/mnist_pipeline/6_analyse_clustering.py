from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load(side="left")

analyser.create_all_reconstrcution_error_plots()
analyser.test_all_decompositions([20, 20, 20, 20])

# analyser.print_shapes()
# analyser.create_cluster_plot(0)

# analyser.print_cluster_infos()
# analyser.create_all_plots()
