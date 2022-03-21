import torch
import lja.utils.config_functions as cfg_funcs


class Manager():
    """The base manager class. Quite minimal. Just has
    things like set up configs, parse args, and anything
    else every manager is going to need."""
    def __init__(self, config_set_name='defaults'):
        super(Manager, self).__init__()

        torch.manual_seed(42)

        self.cfg = cfg_funcs.load_configs(config_set_name)

        # TODO automatically determine device
        if self.cfg.device == 'gpu':
            self.device = torch.device('cuda')
        elif self.cfg.device == 'cpu':
            self.device = torch.device('cpu')