from lja.managers.training_manager import LogicalNetworkTrainingManager
import torch
import torch.nn as nn


def get_stats(jacobian, parameter, y_prime, y):
    num_matches = torch.isclose(jacobian, parameter, atol=1e-2).sum()
    frac_match = num_matches / torch.numel(jacobian)
    print("Fraction matched in jacobian and weight mat: %f" % frac_match.item())

    diff = torch.abs(y - y_prime)
    print("Diff between y and y_prime: %f" % diff.sum().item())


def get_linear_transformations(layer, x):

    # 1. pre-activation
    preact = layer(x)  # [dim_output]
    derivative = torch.heaviside(preact, torch.tensor([0.0]))  # [dim_output]

    # 2. parameters
    params = layer.weight.data
    bias = layer.bias.data

    # 3. Extend for bias
    params_ext = torch.cat((params, bias[:, None]), dim=1)  # [dim_output, dim_input]

    dim_output = params_ext.shape[0]
    dim_input = params_ext.shape[1]
    derivative_ext = derivative[:, None].expand(
        dim_output, dim_input
    )  # [dim_output, dim_input]

    # 4. Calculate linear matrix
    linear_tranformations_ext = torch.multiply(
        params_ext, derivative_ext
    )  # [dim_output, dim_input]

    # 5. Check
    x_ext = torch.cat((x, torch.ones(1)), dim=-1)  # [dim_input]
    y_prime = linear_tranformations_ext @ x_ext  # [dim_output]
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
    x1 = manager.test_loader.dataset[1][0]
    x2 = manager.test_loader.dataset[2][0]

    linear_tranformation = get_linear_transformations(first_layer, x1)
    linear_tranformation = get_linear_transformations(first_layer, x2)
