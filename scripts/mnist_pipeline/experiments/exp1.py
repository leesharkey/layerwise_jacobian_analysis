from lja.managers.training_manager import MnistNetworkTrainingManager
from lja.LT_extractor.extractor import LTExtractor
from lja.decomposition.decomposition import Decomposition
from lja.analyser.analyser import Analyser

n = 100
k = 10

if n > 100 and True:
    raise Exception("Large N: Make sure you have enough space")

# 1. Load model
manager = MnistNetworkTrainingManager(model_type="dropout")

# 2. Input query
# select n-samples of each of the 10 different classes
labels = manager.test_dataset.targets
indices = []
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
del extractor

# 6 decompose
decomp = Decomposition("mnist/dropout/")
decomp.decompose(k, "left").store()
