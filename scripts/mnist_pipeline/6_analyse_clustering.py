from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load(side="left", load_cluster=True)

#
# analyser.print_cluster_shapes()
analyser.reduce_write_matrix_with_cluster_labels(0, 10)
analyser.reduce_write_matrix_with_cluster_labels(1, 10)
analyser.reduce_write_matrix_with_cluster_labels(2, 10)
analyser.reduce_write_matrix_with_cluster_labels(3, 10)
