from lja.analyser.analyser import Analyser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

n = 105
rv = 0

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load_data(side="left", layer=1)
read_vector = analyser.vh[rv, :-1]
print(read_vector.shape)

# now we like to get a linear combination of this by the write vectors of the previous layer
analyser.load_data(side="left", layer=0)
write_vectors = analyser.u[n, :, :]
print(write_vectors.shape)

# use simple linear regression for it
reg = LinearRegression(fit_intercept=False).fit(write_vectors, read_vector)
print(reg.coef_)
print(reg.score(write_vectors, read_vector))

# create input feature based on this combination
feature_vectors = analyser.vh[:, :-1]
print(feature_vectors.shape)

feature_vectors_weighted = np.diag(reg.coef_) @ feature_vectors
feature_vectors_summed = np.sum(feature_vectors_weighted, axis=0)
print(feature_vectors_summed.shape)

# visualize
analyser.plotter.show_plots = True
analyser.plotter.plot_image(feature_vectors_summed.reshape(28, 28))
