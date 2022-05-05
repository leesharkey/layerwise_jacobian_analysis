from lja.analyser.analyser import Analyser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

analyser = Analyser("mnist/dropout/", show_plots=False)

# analyser.analyse_decomposition()
# analyser.print_shapes()

analyser.load_data(side="left", layer=0)
analyser.create_all_plots(n=10, read_threshold=0.35, write_threshold=0.05)

analyser.load_data(side="left", layer=1)
analyser.create_all_plots(n=10, read_threshold=0.05, write_threshold=0.05)

analyser.load_data(side="left", layer=2)
analyser.create_all_plots(n=10, read_threshold=0.05, write_threshold=0.05)
