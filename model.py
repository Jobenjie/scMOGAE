from re import A
import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.nn import Module, Parameter
from layers import Adjacency_Matrix_Dencoder,Encoder1,Expression_Matrix_Decoder
import sys
from fusion import get_fusion_module
from clustering_module import DDC
from loss import Loss


class Mymodel(nn.Module):
    def __init__(self, args):
        super(Mymodel, self).__init__()
    
        self.Lencoder = Encoder1(args.n4,latent_dim=args.n1)            # 图卷积编码器
        self.Gencoder = Encoder1(args.n5,latent_dim=args.n1)

        self.Decoder1 = Expression_Matrix_Decoder(args.n4,latent_dim=args.n1)    # 矩阵解码器
        self.Decoder2 = Expression_Matrix_Decoder(args.n5,latent_dim=args.n1)                  
        
        
        self.Decoder1A = Adjacency_Matrix_Dencoder(latent_dim=args.n1,adj_dim=50)  # 图解码器
        self.Decoder2A = Adjacency_Matrix_Dencoder(latent_dim=args.n1,adj_dim=50)

        self.fusion_inputsize = []
        self.fusion_inputsize.append(self.Lencoder.output_sizes)
        self.fusion_inputsize.append(self.Gencoder.output_sizes)
        self.fusion = get_fusion_module(args, self.fusion_inputsize)      # 加权和模块

        if args.latent_projector_config == False:
            self.latent_projector = nn.Identity()
        else:
            self.latent_projector = nn.Sequential(nn.Linear(args.n1, args.n1,bias=True))


        if args.projector_config == False:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Sequential(nn.Linear(self.fusion.output_size, self.fusion.output_size),nn.ReLU)

        
      

        self.ddc=DDC(input_dim=self.fusion.output_size,args=args)      #聚类模块

        self.loss = Loss(args=args)       # 聚类模块损失和对比损失计算

     

    def pre_train(self,X1, Am1,X2,Am2,sigma1,sigma2):
        X1 = X1+torch.randn_like(X1)*sigma1
        X2 = X2+torch.randn_like(X2)*sigma2         # 随机加入噪声，提升模型鲁棒性
        z_L = self.Lencoder.forward(X1,Am1)
        z_G = self.Gencoder.forward(X2,Am2)

       
        L_pred = self.Decoder1A(z_L)            # 图解码器
        G_pred = self.Decoder2A(z_G)
       

        rec1= self.Decoder1(z_L)       # 矩阵解码器
        rec2= self.Decoder2(z_G)
      
        return z_L,z_G,L_pred,G_pred,rec1,rec2        # 两个模态的编码，图重构，矩阵重构
  


    def forward(self, X1, Am1,X2,Am2,sigma1,sigma2):
        X1 = X1+torch.randn_like(X1)*sigma1
        X2 = X2+torch.randn_like(X2)*sigma2
       
        z_L = self.Lencoder.forward(X1,Am1)
        z_G = self.Gencoder.forward(X2,Am2)


        L_pred = self.Decoder1A(z_L)
        G_pred = self.Decoder2A(z_G)

      
        rec1= self.Decoder1(z_L)
        rec2= self.Decoder2(z_G)

      
        rec1_2= self.Decoder2(z_L)     # 跨模态重构
        rec2_1= self.Decoder1(z_G)

        self.backbone_outputs = [z_L,z_G]
        self.latent_projection = self.latent_projector(torch.cat(self.backbone_outputs, dim=0))
        self.fused = self.fusion(self.backbone_outputs)            # 进行加权和
        self.projections = self.projector(torch.cat(self.backbone_outputs, dim=0))
        self.output, self.hidden = self.ddc(self.fused)         # 通过聚类模块
        loss_values = self.loss(self)

     
        
        return loss_values,z_L,z_G,L_pred,G_pred,rec1,rec2,rec1_2,rec2_1,self.output      # 聚类损失和对比损失相，两个模态的编码，图重构，矩阵重构，跨模态重构，最后的聚类结果
    

    def test(self, X1, Am1,X2,Am2):


        z_L = self.Lencoder.forward(X1,Am1)
        z_G = self.Gencoder.forward(X2,Am2)

        L_pred = self.Decoder1A(z_L)
        G_pred = self.Decoder2A(z_G)

     
        rec1= self.Decoder1(z_L)
        rec2= self.Decoder2(z_G)
      
        rec1_2= self.Decoder2(z_L)
        rec2_1= self.Decoder1(z_G)

        self.backbone_outputs = [z_L,z_G]
        self.fused = self.fusion(self.backbone_outputs)
        self.projections = self.projector(torch.cat(self.backbone_outputs, dim=0))
        self.output, self.hidden = self.ddc(self.fused)
        loss_values = self.loss(self)
        
        return loss_values,z_L,z_G,L_pred,G_pred,rec1,rec2,rec1_2,rec2_1,self.output # 模型测试
    
    def save(self, X1, Am1,X2,Am2):


        z_L = self.Lencoder.forward(X1,Am1)
        z_G = self.Gencoder.forward(X2,Am2)
        
        L_pred = self.Decoder1A(z_L)
        G_pred = self.Decoder2A(z_G)
       
        rec1= self.Decoder1(z_L)
        rec2= self.Decoder2(z_G)
      
        rec1_2= self.Decoder2(z_L)
        rec2_1= self.Decoder1(z_G)

        self.backbone_outputs = [z_L,z_G]
        self.fused = self.fusion(self.backbone_outputs)
        self.projections = self.projector(torch.cat(self.backbone_outputs, dim=0))
        self.output, self.hidden = self.ddc(self.fused)
        loss_values = self.loss(self)

        
        return loss_values,z_L,z_G,L_pred,G_pred,rec1,rec2,rec1_2,rec2_1,self.fused,self.output   # 模型保存，使用self.fused进行可视化展示

