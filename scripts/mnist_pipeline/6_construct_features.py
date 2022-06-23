from lja.feature_constructor.feature_constructor import (
    ConstructorBySample,
    ConstructorByProfile,
)

# By sample
if False:
    constructor = ConstructorBySample(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")
    constructor.set_granularity("profile")

    constructor.construct_single_feature(
        layer=1,
        feature_index=0,
        target_index=5,
        plot=True,
        store=False,
        reuse_stored_features=False,
    )

if False:

    constructor = ConstructorBySample(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")

    constructor.construct_multiple_features(
        layers=[1, 2],
        feature_indices=[0, 1],
        target_indices=[0, 1, 101, 102, 201, 202, 901, 902,],
        granularites=["sample", "profile"],
        plot=True,
        store=True,
    )

# By Profile
if True:
    constructor = ConstructorByProfile(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")
    constructor.set_granularity("profile")

    constructor.construct_single_feature(
        layer=2,
        feature_index=0,
        target_index=5,
        plot=True,
        store=True,
        reuse_stored_features=False,
    )

if False:
    constructor = ConstructorByProfile(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")

    constructor.construct_multiple_features(
        layers=[1, 2],
        feature_indices=[0, 1],
        target_indices=[0, 1, 101, 102],
        granularites=["profile"],
        plot=True,
        store=True,
    )
