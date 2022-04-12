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
    preact = layer(x)
    derivative = torch.heaviside(preact, torch.tensor([0.]))

    # 2. parameters
    params = layer.weight.data
    bias = layer.bias.data

    # 3. Extend for bias
    params_ext = torch.cat((params, bias[:, None]), dim=1)
    in_size = params_ext.shape[1]
    out_size = params_ext.shape[0]
    derivative_ext = derivative[:, None].expand(out_size, in_size)

    # 4. Calculate linear matrix
    linear_tranformations_ext = torch.multiply(params_ext, derivative_ext)

    # 5. Check
    x_ext = torch.cat((x, torch.ones(1)), dim=-1)
    y_prime = linear_tranformations_ext @ x_ext
    act = nn.ReLU(inplace=True)
    y = act(preact)
    get_stats(linear_tranformations_ext, params_ext, y_prime, y)

    # TODO: expand to multiple data points!!!

    return linear_tranformations_ext


if __name__ == "__main__":

    # 1. Load model
    manager = LogicalNetworkTrainingManager()
    model = manager.net

    # 2. Extract first layer
    first_layer = model.nets[0]

    # 3. Get Linear Transformation
    x = manager.test_loader.dataset[1][0]
    linear_tranformation = get_linear_transformations(first_layer, x)
