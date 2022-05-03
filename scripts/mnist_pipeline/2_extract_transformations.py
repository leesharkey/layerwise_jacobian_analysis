from lja.managers.training_manager import MnistNetworkTrainingManager
from lja.LT_extractor.extractor import LTExtractor
import torch

# 1. Load model
manager = MnistNetworkTrainingManager(model_type="dropout")
manager.validation_loop()

# 2. Input query
# select n-samples of each of the 10 different classes
labels = manager.test_dataset.targets
indices = []
n = 3
for i in range(10):
    index = (labels == i).nonzero(as_tuple=True)[0][0:n]
    indices += index.tolist()

# select final input query
x0 = manager.test_dataset.data.reshape(-1, 28 * 28).float()[indices, :]
labels = labels[indices]

# 3. Create extractor
extractor = LTExtractor(manager.net, x0, labels)

# 4. Extract linear transformations
extractor.extract()

# 5. Store
extractor.store("mnist/dropout/")
