from lja.managers.training_manager import LogicalNetworkTrainingManager
from lja.LT_extractor.extractor import LTExtractor
import torch

# 1. Load model
manager = LogicalNetworkTrainingManager(problem="stacked_xor")

# 2. Input query
x0 = torch.tensor(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=torch.float,
)

# 3. Create extractor
extractor = LTExtractor(manager.net, x0)

# 4. Extract linear transformations
extractor.extract()

# 5. Store
extractor.store("logical/stacked_xor/")
