from lja.feature_constructor.feature_constructor import (
    ConstructorBySample,
    ConstructorByProfile,
)
import time
import numpy as np

# By sample
constructor = ConstructorBySample(path="mnist/dropout/", show_plots=False)
constructor.load()

# Params
constructor.set_k_per_layer([20, 10, 10, 10])
constructor.set_granularity("sample")


if False:
    start = time.time()
    constructor.construct_multiple_features(
        layers=[1, 2, 3],
        feature_indices=range(10),
        target_indices=range(1000),
        granularites=["sample"],
        plot=False,
        store=True,
    )
    end = time.time()
    print("The time of execution of above program is :", end - start)

if True:
    start = time.time()
    constructor.construct_multiple_features(
        layers=[0, 1, 2, 3],
        feature_indices=range(10),
        target_indices=[352, 322],
        granularites=["sample"],
        plot=True,
        store=False,
    )
    end = time.time()
    print("The time of execution of above program is :", end - start)
