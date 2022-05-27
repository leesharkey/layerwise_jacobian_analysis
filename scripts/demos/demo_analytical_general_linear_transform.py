import torch
import torch.nn as nn
from functools import partial


def get_scaled_transform_matrix(postactivations, transformation_mat, preactivations):
    scaling_vec = postactivations / preactivations
    new_transform_mat = torch.multiply(
        transformation_mat.transpose(0, 1), scaling_vec
    ).transpose(0, 1)
    return new_transform_mat


def get_stats(transform_matrix, parameter, y_prime, y):
    num_matches = torch.isclose(transform_matrix, parameter, atol=5e-3).sum()
    frac_match = num_matches / torch.numel(transform_matrix)
    print("Fraction matched in transform matrix and weight matrix: %f" % frac_match.item())

    diff = torch.abs(y - y_prime)
    print("Diff between y and y_prime: %f" % diff.sum().item())


if __name__ == "__main__":

    activation = "softmax"

    if activation == "relu":
        act = nn.ReLU(inplace=True)

    elif activation == "leaky":
        act = nn.LeakyReLU(inplace=True)

    elif activation == "gelu":
        act = torch.nn.functional.gelu

    elif activation == "sigmoid":
        act = nn.Sigmoid()

    elif activation == "softmax":
        act = partial(torch.nn.functional.softmax, dim=0)

    else:
        print("Invalid activation")

    size = 1000
    shift = 0
    device = "cpu"
    net = nn.Linear(in_features=size, out_features=size, bias=True).to(device)
    net.state_dict()["weight"] += shift

    # input/ output
    x = torch.randn(size).to(device) + shift
    x = x.requires_grad_()

    # activations
    preact = net(x)
    y = act(preact)

    # parameters
    params = net.weight.data
    bias = net.bias.data

    # 1. Extended approach
    x_ext = torch.cat((x, torch.ones(1)), dim=-1)
    params_ext = torch.cat((params, bias[:, None]), dim=1)
    preact_ext = params_ext @ x_ext

    linear_mat = get_scaled_transform_matrix(y, params_ext, preact_ext)
    y_prime = linear_mat @ x_ext

    get_stats(linear_mat, params_ext, y_prime, y)
