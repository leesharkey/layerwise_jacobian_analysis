from lja.managers.training_manager import LogicalNetworkTrainingManager
import torch
import torch.nn as nn


def get_stats(jacobian, parameter, y_prime, y):

    num_matches = torch.isclose(jacobian, parameter, atol=1e-2).sum(dim=(1, 2))
    frac_match = num_matches / torch.numel(jacobian[0])
    print("Min Fraction matched:", frac_match.min())
    print("Max Fraction matched:", frac_match.max())
    print(
        "Average Fraction matched in jacobian and weight mat: .%f" % frac_match.mean()
    )

    diff = torch.abs(y - y_prime).sum(dim=1)
    print("Max Differences:", diff.max().item())
    print("Mean diff between y and y_prime: %f" % diff.mean())


def get_linear_transformations(layer, x):

    # 1. pre-activation
    preact = layer(x)  # [n, dim_output]
    derivative = torch.heaviside(preact, torch.tensor([0.0]))  # [n, dim_output]

    # 2. parameters
    params = layer.weight.data
    bias = layer.bias.data

    # 3. Extend for bias
    params_ext = torch.cat((params, bias[:, None]), dim=1)  # [dim_output, dim_input]

    n = x.shape[0]
    dim_output = params_ext.shape[0]
    dim_input = params_ext.shape[1]
    derivative_ext = derivative[:, :, None].expand(
        -1, dim_output, dim_input
    )  # [n, dim_output, dim_input]

    # 4. Calculate linear matrix
    linear_tranformations_ext = torch.multiply(
        params_ext, derivative_ext
    )  # [n, dim_output, dim_input]

    # 5. Check
    x_ext = torch.cat((x, torch.ones(n, 1)), dim=-1)  # [n, dim_input]
    x_ext = x_ext[:, None, :].reshape((n, dim_input, 1))  # [n, dim_input, 1]
    y_prime = torch.squeeze(
        torch.bmm(linear_tranformations_ext, x_ext)
    )  # [n, dim_output]
    act = nn.ReLU(inplace=True)
    y = act(preact)

    get_stats(linear_tranformations_ext, params_ext, y_prime, y)

    return linear_tranformations_ext


if __name__ == "__main__":

    # 1. Load model
    manager = LogicalNetworkTrainingManager()
    model = manager.net

    # 2. Extract first layer
    first_layer = model.nets[0]

    # 3. Get Linear Transformation
    x = manager.test_loader.dataset[:][0]

    linear_tranformation = get_linear_transformations(first_layer, x)
