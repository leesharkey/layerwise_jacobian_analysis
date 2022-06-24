from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load(side="left")

# decompsotion testing
analyser.create_all_reconstrcution_error_plots()
analyser.create_all_singluarvalue_plots()
analyser.test_all_decompositions([100, 100, 100, 10])

# general information
analyser.print_shapes()

# embedding
# analyser.create_all_reductions_plots()
