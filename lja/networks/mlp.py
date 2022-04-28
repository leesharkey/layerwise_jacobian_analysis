import torch
import torch.nn as nn


class NLayerPerceptron(nn.Module):
    """An N layer perceptron with a linear output."""

    def __init__(
        self,
        sizes=None,
        last_act=nn.Softmax,
        device=torch.device("cuda"),
        dropout=False,
    ):
        super(NLayerPerceptron, self).__init__()
        self.nets = nn.ModuleList([])
        self.act = nn.ReLU(inplace=True)
        self.device = device
        self.dropout = dropout
        self.dropout_net = nn.Dropout(p=0.5, inplace=False)

        if last_act is not None:
            self.last_act = last_act()
        else:
            self.last_act = None
        self.num_layers = len(sizes) - 1

        for i in range(self.num_layers):
            net = nn.Linear(in_features=sizes[i], out_features=sizes[i + 1]).to(
                self.device
            )
            self.nets.append(net)

    def forward(self, x, return_all_activations=False):
        outs = []
        for layer_idx, net in enumerate(self.nets):
            x = net(x)

            if layer_idx < self.num_layers - 1:  # No act on last layer
                if self.dropout:
                    x = self.dropout_net(x)
                x = self.act(x)
            elif layer_idx == self.num_layers - 1:
                if self.last_act is not None:
                    x = self.last_act(x)
            else:
                ValueError(f"layer_idx shouldn't have this value: {layer_idx}")

            if return_all_activations:
                outs.append(x)

        out = x
        if out.shape[-1] == 1:  # as in logical circuit
            out = torch.squeeze(out)

        if return_all_activations:
            return out, outs
        else:
            return out

    def get_layer_and_act(self, layer_key):

        layer = self.nets[layer_key]

        if layer_key < self.num_layers - 1:
            act = self.act
        elif layer_key == self.num_layers - 1:
            act = self.last_act
        else:
            NotImplementedError("Inappropriate layer key.")
        return layer, act

    def get_depth(self):
        return len(self.nets)
