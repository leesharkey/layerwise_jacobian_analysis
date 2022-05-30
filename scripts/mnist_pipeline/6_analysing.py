from lja.analyser.analyser import Analyser

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load_data(side="left", layer=0)

analyser.create_all_plots()
