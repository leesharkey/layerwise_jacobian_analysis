from lja.feature_constructor.feature_constructor import (
    ConstructorBySample,
    ConstructorByProfile,
    ConstructorByProfileCluster,
)

# By sample
if True:
    constructor = ConstructorBySample(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")

    constructor.construct_single_feature(
        layer=2,
        feature_index=0,
        target_index=5,
        plot=True,
        store=True,
        reuse_stored_features=False,
    )

    constructor.construct_multiple_features(
        layers=[1, 2],
        feature_indices=[0, 1],
        target_indices=[0, 1, 101, 102, 201, 202, 901, 902,],
        granularites=["sample", "profile", "profile_cluster"],
        plot=True,
        store=True,
    )

# By Profile
if False:
    constructor = ConstructorByProfile(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")

    constructor.construct_single_feature(
        layer=2,
        feature_index=0,
        target_index=5,
        plot=True,
        store=True,
        reuse_stored_features=False,
    )

    constructor.construct_multiple_features(
        layers=[1, 2],
        feature_indices=[0, 1],
        target_indices=[0, 1, 101, 102],
        granularites=["profile", "profile_cluster"],
        plot=True,
        store=True,
    )

# By Profile Cluster
if False:
    constructor = ConstructorByProfileCluster(
        path="mnist/dropout_feature/", show_plots=False
    )
    constructor.load_data(load_path="mnist/dropout/", side="left")

    constructor.construct_single_feature(
        layer=2,
        feature_index=0,
        target_index=5,
        plot=True,
        store=True,
        reuse_stored_features=False,
    )

    constructor.construct_multiple_features(
        layers=[1, 2],
        feature_indices=[0, 1],
        target_indices=[0, 1, 2, 3, 4, 5, 6],
        granularites=["profile_cluster"],
        plot=True,
        store=True,
    )
