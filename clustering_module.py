import torch.nn as nn


class none_layer(nn.Module):     # 聚类模块构建
    def __init__(self):
        
        super().__init__()
        
        self.output = None
    
    def forward(self, x):
        self.output = x
        return self.output

class DDC(nn.Module):
    def __init__(self, input_dim,args):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()

        hidden_layers = [nn.Linear(input_dim[0], args.ddc_hidden), nn.ReLU()]
        if args.ddcuse_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=args.ddc_hidden))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(args.ddc_hidden, args.n_clusters), nn.Softmax(dim=1))
        if args.ddc_direct:
            self.hidden = none_layer()
            self.output = nn.Sequential(nn.Linear(input_dim[0], args.n_clusters), nn.Softmax(dim=1))


    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden
