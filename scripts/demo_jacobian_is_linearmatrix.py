import torch
import torch.nn as nn

size = 7
shift = 0
device = "cuda"
net = nn.Linear(in_features=size, out_features=size, bias=False).to(device)
net.state_dict()["weight"] += shift
act = nn.LeakyReLU(inplace=True)

if __name__ == "__main__":

    x = torch.randn(size).to(device) + shift
    x = x.requires_grad_()
    y = act(net(x))
    net_func = lambda inp: act(net(inp))

    ### This line......
    jacobian = torch.autograd.functional.jacobian(net_func, x, create_graph=True)

    ### ... is equivalent to these lines:
    # grads = []
    # for i in range(size):
    #     y[i].backward(retain_graph=True)
    #     grads.append(x.grad.data.clone())
    #     x.grad.data = torch.zeros_like(x.grad.data)
    # jacobian = torch.stack(grads)

    num_matches = torch.isclose(
        [p for p in net.parameters()][0], jacobian, atol=1e-2
    ).sum()
    num_total = size**2
    frac_match = num_matches / num_total
    print("Fraction matched in jacobian and weight mat: %f" % frac_match.item())

    y_prime = jacobian @ x
    diff = torch.abs(y - y_prime)
    print("Diff between y and y_prime: %f" % diff.sum().item())
