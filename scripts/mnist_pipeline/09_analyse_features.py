from lja.analyser.analyser import Analyser
import numpy as np

# from lja.managers.training_manager import MnistNetworkTrainingManager
# import torch

analyser = Analyser("mnist/dropout/", show_plots=False)
analyser.load(side="left", load_features=True)
analyser.plotter.plot_path = "plots/features/"
analyser.set_k_per_layer([20, 10, 10, 10])

if False:
    np.save("results/feature.npy", analyser.data.feature_list)

# Crate emebdding plots for features
if False:
    # analyser.create_all_feature_reduction_plots()
    analyser.create_all_feature_average_plots()

# Reconstruction Score
if True:
    # analyser.test_features(baseline=True)
    analyser.test_features()

# Reconstruction Score with validation set
if False:
    # 1. Load model
    manager = MnistNetworkTrainingManager(model_type="dropout")

    # select n-samples of each of the 10 different classes
    labels = manager.test_dataset.targets
    indices = []
    n = 10
    for i in range(10):
        index = (labels == i).nonzero(as_tuple=True)[0][100 : (100 + n)]
        indices += index.tolist()

    # select input query from test dataset
    x0 = manager.test_dataset.data.float().reshape(-1, 28 * 28)[indices, :]
    labels = labels[indices]

    # get activations
    activations = [x0]
    x = x0
    for i in range(4):
        layer, act = manager.net.get_layer_and_act(i)
        x = act(layer(x))
        activations.append(x.detach().numpy())

    analyser.test_features_with_validation(x0.numpy(), activations, labels)
