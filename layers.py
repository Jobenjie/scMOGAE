import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# import dgl
#from torch_geometric.nn import TAGConv
# from spektral.layers import GraphAttention, GraphConvSkip

sp = nn.Softplus()
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
DispAct = lambda x: torch.clamp(sp(x), 1e-4, 1e4)

class InnerProductDecoder(nn.Module):            # 图解码器组件
    def __init__(self, adj_dim):
        super(InnerProductDecoder, self).__init__()
        # 32 * 32
        self.weight = nn.Parameter(torch.Tensor(adj_dim, adj_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # h1 = x * kernel
        # output = h1 * x.t
        h = torch.matmul(x, self.weight)
        output = torch.matmul(h, x.T)
        return output

class Adjacency_Matrix_Dencoder(nn.Module):           #图解码器
    def __init__(self, latent_dim=15, adj_dim=32):
        super(Adjacency_Matrix_Dencoder, self).__init__()
        self.dense = nn.Linear(latent_dim, adj_dim)
        self.bilinear = InnerProductDecoder(adj_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        # dec_in = n * latent_dim
        # h = L(dec_in) : n * adj_dim
        # h = BL(h) : n * n
        # h = sigmod(h) : n * n
        x = self.dense(x)
        x = self.bilinear(x)
        x = self.sigmod(x)
        return x
  

class Encoder1(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, latent_dim=15):         # 图卷积编码器
        super(Encoder1, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=0.2)
        self.conv = my_GCN(in_channels, hidden_dim, use_act=True)
        self.conv2 = my_GCN(hidden_dim, latent_dim, use_act=False)

    @property
    def output_sizes(self):
        return self.latent_dim
    def forward(self, x, adj):
        # x = dropout(x) : n*indim
        # x = GCN(x, a) : n*hidden_dim
        # x = relu(x) : n*hidden_dim
        # x = GCN(x, a) : n*latent_dim
        
        x = self.conv(x, adj)
        x = self.dropout(x)
        x = self.conv2(x, adj)

        return x
class my_GCN(nn.Module):                                  # 图卷积网络
    def __init__(self, in_features, out_features, use_act = True):
        super(my_GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_act = use_act
        if self.use_act:
            self.act = F.relu
        # 编码器
        self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x.float()
        adj = adj.float()
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        x = x + self.bias
        if self.use_act:
            x = self.act(x)
        return x
 
class Expression_Matrix_Decoder(nn.Module):          # 矩阵解码器
    def __init__(self, in_channels, latent_dim=15, dec_dim=[128, 256, 512],use_act = True):
        super(Expression_Matrix_Decoder, self).__init__()
        layers = []
        for i in range(len(dec_dim)):
            if i == 0:
                # decx_in : n * latent_dim
                # h = L(decx_in) : n * dec_dim[0]
                # h = relu(h) : n * dec_dim[0]
                layers.append(nn.Linear(latent_dim, dec_dim[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dec_dim[i-1], dec_dim[i]))
                layers.append(nn.ReLU())
        self.multilayer = nn.Sequential(*layers)
        self.dense_f = nn.Linear(dec_dim[-1], in_channels)
        self.activation_flag = use_act
        if self.activation_flag :
            self.activation_f = nn.Sigmoid()

    def forward(self, x):
        x = self.multilayer(x)
        x = self.dense_f(x)
        if self.activation_flag :
            x = self.activation_f(x)
        return x

 
class FC_Classifier(nn.Module):                # 组学标签判别器
    """Latent space discriminator"""
    def __init__(self, nz, n_hidden=50, n_out=1):
        super(FC_Classifier, self).__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_hidden, 2*n_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*n_hidden,n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)



