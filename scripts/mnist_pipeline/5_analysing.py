from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load_data(side="left")


analyser.print_shapes()
analyser.create_all_plots()
