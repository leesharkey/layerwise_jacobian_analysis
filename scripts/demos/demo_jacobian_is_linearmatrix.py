import torch
import torch.nn as nn

def leaky_derivative(input):
    leaky_slope = 0.01
    derivative = torch.heaviside(preact, torch.tensor([leaky_slope]))
    derivative[derivative==0] = leaky_slope

    return derivative

def relu_derivative(input):
    return torch.heaviside(input, torch.tensor([0.]))

def sigmoid_derivative(input):
    return torch.multiply(torch.sigmoid(input), (1-torch.sigmoid(input)))

def get_stats(jacobian, parameter, y_prime, y):
    num_matches = torch.isclose(jacobian, parameter, atol=1e-2).sum()
    frac_match = num_matches / torch.numel(jacobian)
    print("Fraction matched in jacobian and weight mat: %f" % frac_match.item())

    diff = torch.abs(y - y_prime)
    print("Diff between y and y_prime: %f" % diff.sum().item())


size = 400
shift = 0
device = "cpu"
net = nn.Linear(in_features=size, out_features=size, bias=True).to(device)
net.state_dict()["weight"] += shift
activation = 'relu'


if activation == 'relu':
    act = nn.ReLU(inplace=True)
    derivative_function = relu_derivative

elif activation == 'leaky':
    act = nn.LeakyReLU(inplace=True)
    derivative_function = leaky_derivative

elif activation == 'sigmoid':
    act = nn.Sigmoid()
    derivative_function = sigmoid_derivative

else:
    print('Invalid activation')



if __name__ == "__main__":

    # input/ output
    x = torch.randn(size).to(device) + shift
    x = x.requires_grad_()
    y = act(net(x))

    # activations
    preact = net(x)
    derivative = derivative_function(preact)

    # parameters
    params = net.weight.data
    bias = net.bias.data


    # 1. extended Aproach
    x_ext = torch.cat((x, torch.ones(1)), dim=-1)
    params_ext = torch.cat((params, bias[:, None]), dim=1)
    preact_ext = params_ext @ x_ext

    derivative_ext = derivative[:, None].expand(size, size+1)
    jacobian_ext = torch.multiply(params_ext, derivative_ext)
    y_prime = jacobian_ext @ x_ext

    get_stats(jacobian_ext, params_ext, y_prime, y)


    # 2. split approach
    jacob_w = torch.multiply(derivative[:, None], params)

    # jacobian == jacob_w
    net_func = lambda inp: act(net(inp))
    jacobian = torch.autograd.functional.jacobian(net_func, x, create_graph=True)

    y_prime = jacob_w @ x + torch.multiply(derivative, bias)
    get_stats(jacob_w, params, y_prime, y)
