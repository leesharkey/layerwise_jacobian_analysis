import torch


def relu_derivative(input):
    return torch.heaviside(input, torch.tensor([0.0]))


def sigmoid_derivative(input):
    return torch.multiply(torch.sigmoid(input), (1 - torch.sigmoid(input)))


def leaky_derivative(input):
    leaky_slope = 0.01
    derivative = torch.heaviside(input, torch.tensor([leaky_slope]))
    derivative[derivative == 0] = leaky_slope

    return derivative


def get_derivative(act):

    print(act.__class__.__name__)
    if act.__class__.__name__ == "ReLU":
        return relu_derivative
    elif act.__class__.__name__ == "Sigmoid":
        return sigmoid_derivative
    elif act.__class__.__name__ == "LeakyReLU":
        return leaky_derivative
    elif act.__class__.__name__ == "Softmax":
        return relu_derivative
    else:
        raise Exception("activation_derivatives: derivative not implemented")
