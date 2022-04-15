import torch
import torch.utils.data as data


class LogicalDataGenerator(data.Dataset):
    """
    Yoinked from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#Learning-by-example:-Continuous-XOR
    then modified.
    """

    def __init__(self, size, problem="stacked_xor", std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std

        if problem == "xor":
            self.logical_circuit = self.xor
            data, label = self.generate_xor()
        elif problem == "stacked_xor":
            self.logical_circuit = self.stacked_xor
            data, label = self.generate_stacked_xor()
        else:
            raise Exception("LogicalDataGenerator: invalid problem")

        self.data = data
        self.label = label

    def xor(self, inp):
        assert inp.shape[1] == 2
        return inp.sum(dim=1) == 1

    def stacked_xor(self, inp):
        xor1 = self.xor(inp[:, 0:2])
        xor2 = self.xor(inp[:, 2:])
        inp = torch.stack([xor1, xor2], dim=-1)
        assert inp.shape[1] == 2
        return self.xor(inp)

    def generate_stacked_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 4), dtype=torch.float32)
        label = (self.stacked_xor(data.int())).to(torch.long)
        return data, label

    def generate_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (self.xor(data.int())).to(torch.long)
        return data, label

    def generate_stacked_continuous_xor(self):

        data = torch.randint(low=0, high=2, size=(self.size, 4), dtype=torch.float32)
        label = (self.stacked_xor(data.int())).to(torch.long)
        data += self.std * torch.randn(data.shape)
        return data, label

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)
        return data, label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
