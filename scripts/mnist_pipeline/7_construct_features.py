from lja.feature_constructor.feature_constructor import (
    ConstructorBySample,
    ConstructorByProfile,
)

# By sample
if True:
    constructor = ConstructorBySample(path="mnist/dropout/", show_plots=False)
    constructor.load()
    constructor.set_granularity("sample")

    constructor.construct_multiple_features(
        layers=[1, 2, 3],
        feature_indices=[0, 1],
        target_indices=[0, 1, 101, 102, 201, 202, 901, 902],
        granularites=["sample"],
        plot=True,
        store=True,
    )


if False:
    constructor = ConstructorBySample(path="mnist/dropout_feature/", show_plots=False)
    constructor.load_data(load_path="mnist/dropout/", side="left")

    constructor.construct_single_feature(
        layer=3,
        feature_index=0,
        target_index=4,
        plot=True,
        store=True,
        reuse_stored_features=True,
    )

    constructor.construct_multiple_features(
        layers=[1, 2],
        feature_indices=[0, 1],
        target_indices=[0, 1, 101, 102, 201, 202, 901, 902,],
        granularites=["sample", "profile"],
        plot=True,
        store=True,
    )
