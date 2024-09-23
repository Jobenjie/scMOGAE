from utils import clustering,eva_model,save_result
from loss import A_recloss
from layers import FC_Classifier
from process_data import normalize2
from model import Mymodel
import anndata as ad
import os
import math
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from torch import optim
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
from sklearn.preprocessing import LabelEncoder
from graph_funcion import get_adj,get_adj_batch
from utils import valid
from spektral.layers import GraphConv
from itertools import chain
import random
import warnings
import pickle
warnings.filterwarnings("ignore")
encoder = LabelEncoder()


# 矩阵重构损失

def Pretrain_Loss(x, p):
    return -torch.mean(x * torch.log(torch.clamp(p, min=1e-12, max=1.0)))

#  模型预训练
def pre_train(model, netClf,args, X1, y1, adj1,adjn1,X2, y2, adj2,adjn2,device):
    X1 = X1.float()
    X2 = X2.float()
    adjn1 = adjn1.float()
    adjn2 = adjn2.float()
    netClf=netClf.to(device)      # 判别器
    optimizer = optim.Adam(model.parameters(), lr=args.lr1)   # 主模型的优化器
    opt_netClf = optim.Adam(netClf.parameters(), lr=args.lr1)   # 判别器的优化器
    criterion_dis=torch.nn.MSELoss()
    
    for epoch in range(1, 1 + args.epochs1):
        model.train()
        netClf.eval()
        z_L,z_G,L_pred,G_pred,rec1,rec2= model.pre_train(X1, adjn1, X2, adjn2,args.sigma1,args.sigma2)

        rec_A_loss1 = A_recloss(adj1, L_pred)    # 图重构损失
        rec_A_loss2 = A_recloss(adj2, G_pred)

        pre_loss1 = Pretrain_Loss(X1,rec1)     # 矩阵重构损失
        pre_loss2 = Pretrain_Loss(X2,rec2)

        rna_scores = netClf(z_L)              # 判别器损失
        atac_scores = netClf(z_G)
        rna_scores=torch.squeeze(rna_scores,dim=1)
        atac_scores=torch.squeeze(atac_scores,dim=1)
        rna_labels = torch.zeros(rna_scores.size(0),).float().to(device)
        atac_labels = torch.ones(atac_scores.size(0),).float().to(device)
        clf_loss = 0.5*criterion_dis(rna_scores, atac_labels) + 0.5*criterion_dis(atac_scores, rna_labels)

        loss = torch.mean(0.01*rec_A_loss1 +  pre_loss1 + 0.01*rec_A_loss2 +  pre_loss2+clf_loss)  # 损失加和
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('pre_train_loss_1:')
            print(
                str(epoch) + "   " + str(loss.item()) + "   " + str(torch.mean(pre_loss1 + pre_loss2).item()) + "  " +
                str(torch.mean(rec_A_loss1 + rec_A_loss2).item())+ "  " +str(torch.mean(clf_loss).item()))
        

        model.eval()
        netClf.train()         # 固定主模型，训练判别器

        z_L,z_G,L_pred,G_pred,rec1,rec2 = model.pre_train(X1, adjn1, X2, adjn2,args.sigma1,args.sigma2)
        rna_scores = netClf(z_L)
        atac_scores = netClf(z_G)
        rna_labels = torch.zeros(rna_scores.size(0),).float().to(device)
        atac_labels = torch.ones(atac_scores.size(0),).float().to(device)
        rna_scores=torch.squeeze(rna_scores,dim=1)
        atac_scores=torch.squeeze(atac_scores,dim=1)
        clf_loss = 0.5*criterion_dis(rna_scores, rna_labels) + 0.5*criterion_dis(atac_scores, atac_labels)  # 判别器损失
        loss1 = clf_loss
        
        opt_netClf.zero_grad()      # 优化判别器
        loss1.backward()
        opt_netClf.step()


        epoch += 1
        if epoch % 10 == 0:
            print('pre_train_loss_2:')
            print(str(epoch) + "   " + str(loss1.item()))
    
    nmi1, ari1, _ = clustering(args, z_L, y1, adj1)   # 使用k-means聚类和谱聚类进行评估的隐编码是否有效，输出最好的结果
    nmi2, ari2, _ = clustering(args, z_G, y2, adj2)
    
    print("NMI1: {:.4f},".format(nmi1), "ARI1: {:.4f},".format(ari1))
    print("NMI2: {:.4f},".format(nmi2), "ARI2: {:.4f},".format(ari2))

    print("pre_train over")



# 正式训练开始
def alt_train(model,netClf, args, X1, y1, adj1,adjn1,X2, y2, adj2,adjn2,Nsample,device):
    print('-------------train start--------------------')
    X1 = X1.float()
    X2 = X2.float()
    adjn1 = adjn1.float()
    adjn2 = adjn2.float()
    netClf=netClf.to(device)
 
    parameters1 = chain.from_iterable(        # 制作需要优化的参数集合
                [
                    model.fusion.parameters(),
                    model.latent_projector.parameters(),
                    model.projector.parameters(),
                    model.ddc.parameters(),
                ]
            )
    parameters2 = chain.from_iterable(
                [
                    model.Lencoder.parameters(),
                    model.Gencoder.parameters(),
                    model.Decoder1.parameters(),
                    model.Decoder2.parameters(),
                    model.Decoder1A.parameters(),
                    model.Decoder2A.parameters(),
                      
                ]
            )
    optimizer1 = optim.Adam(parameters1, lr=args.lr)   # 因为主模型是多任务训练，需要两个优化器
    optimizer2 = optim.Adam(parameters2, lr=args.lr)   
   
    opt_netClf = optim.Adam(netClf.parameters(), lr=args.lr)    # 判别器优化器
    criterion_dis=torch.nn.MSELoss()

    batch = args.batch
    for epoch in range(1, 1 + args.epochs2):
       
        rec_A_loss_all1 = 0   # 图重构的总损失
        
        pre_loss_all1 = 0   # 聚类任务的矩阵重构损失
        pre_loss_all2 = 0  # 跨模态任务的矩阵重构损失
        corpre_loss_all = 0   # 跨模态任务的矩阵重构损失

        _,_,_,_,_,_,_=work1(model,netClf, X1, adj1,adjn1,X2, adj2,adjn2,optimizer1,criterion_dis,args.sigma1,args.sigma2)   # 先执行3次聚类任务，并且进行优化
        _,_,_,_,_,_,_=work1(model,netClf, X1, adj1,adjn1,X2, adj2,adjn2,optimizer1,criterion_dis,args.sigma1,args.sigma2)
        rec_A_loss1,rec_A_loss2,pre_loss1,pre_loss2,loss_ddc,clf_loss,loss_total1=work1(model,netClf, X1, adj1,adjn1,X2, adj2,adjn2,optimizer1,criterion_dis,args.sigma1,args.sigma2)


        rec_A_loss_all1 += rec_A_loss1.detach().item() + rec_A_loss2.detach().item()   
        pre_loss_all1 += pre_loss1.detach().item() + pre_loss2.detach().item()
 

        if epoch % 10 == 0 :
            print('alt_train_loss_clus:')
            print(str(epoch) + "   " + str(loss_total1.detach().item()) + "   " + str(pre_loss_all1) + "  " + str(rec_A_loss_all1) + "  "+ str(loss_ddc.detach().item() )+ "  "+str( clf_loss.detach().item()))


        pre_loss11 ,pre_loss22,loss_con,loss_total2=work2(model,netClf, X1, adj1,adjn1,X2, adj2,adjn2,optimizer2,args.sigma1,args.sigma2)   # 再执行1次跨模态任务，并进行优化

       
        corpre_loss_all += pre_loss11.detach().item() + pre_loss22.detach().item()
        
        if epoch % 10 == 0 :
            print('alt_train_loss_tran:')
            print(str(epoch) + "   " + str(loss_total2.detach().item()) + "  "+ str(corpre_loss_all)+"  "+str(loss_con))



        model.eval()
        netClf.train()          # 同样的，我们对判别器进行优化

        _,z_L,z_G,_,_,_,_,_,_,_= model(X1, adjn1, X2, adjn2,args.sigma1,args.sigma2)
        rna_scores = netClf(z_L)
        atac_scores = netClf(z_G)
        rna_labels = torch.zeros(rna_scores.size(0),).float().to(device)
        atac_labels = torch.ones(atac_scores.size(0),).float().to(device)
        rna_scores=torch.squeeze(rna_scores,dim=1)
        atac_scores=torch.squeeze(atac_scores,dim=1)
        disclf_loss = 0.5*criterion_dis(rna_scores, rna_labels) + 0.5*criterion_dis(atac_scores, atac_labels)
        del rna_scores,atac_scores,rna_labels,atac_labels
        loss1 = disclf_loss

        opt_netClf.zero_grad()
        loss1.backward()
        opt_netClf.step()
      
            
   
        if epoch % 10 == 0 :
            print('alt_train_loss_dis:')
            print(str(epoch) + "   " + str(disclf_loss.detach().item() ))
        if epoch>=150 and epoch % 10 == 0:

            res=eva_model(model,netClf, X1,adjn1,X2,adjn2,y1,Nsample,args.device)
            if epoch == 660:                                        
                save_result(model,netClf, X1,adjn1,X2,adjn2,y1,Nsample,args.device,args.save_dir)              # 评估训练结果
                model_file = os.path.join(args.save_dir, 'my_model.pkl')
                with open(model_file, 'wb') as f:                                                              # 保存模型
                    pickle.dump(model, f)
            for key, value in res.items():
                if key != "cmat" and key != "acc":
                    print(key + ": ", value)
                

        epoch += 1
       

        
def work1(model,netClf, X1, adj1,adjn1,X2, adj2,adjn2,optimizer1,criterion_dis,sigma1,sigma2):
    model.train()
    netClf.eval()
    loss_values,z_L,z_G,L_pred,G_pred,rec1,rec2,_,_,_,= model(X1, adjn1, X2, adjn2,sigma1,sigma2)
    
    rec_A_loss1 = A_recloss(adj1, L_pred)              # 两个图重构损失
    rec_A_loss2 = A_recloss(adj2, G_pred)

    pre_loss1 = Pretrain_Loss(X1,rec1)                 # 两个矩阵重构损失
    pre_loss2 = Pretrain_Loss(X2,rec2)

    rna_scores = netClf(z_L)                     # 判别器损失
    atac_scores = netClf(z_G)
    rna_scores=torch.squeeze(rna_scores,dim=1)
    atac_scores=torch.squeeze(atac_scores,dim=1)
    rna_labels = torch.zeros(rna_scores.size(0),).float().to(device)
    atac_labels = torch.ones(atac_scores.size(0),).float().to(device)
    clf_loss = 0.5*criterion_dis(rna_scores, atac_labels) + 0.5*criterion_dis(atac_scores, rna_labels)

    del rna_scores,atac_scores,rna_labels,atac_labels

    loss_clu = loss_values.get("SelfEntropy")+loss_values.get("ddcs")               # 从loss_values中提取聚类模块的损失
    loss_total1 = torch.mean(0.01*rec_A_loss1 +  10*pre_loss1 + 0.01*rec_A_loss2 +  10*pre_loss2+loss_clu*0.005+clf_loss)
    optimizer1.zero_grad()
    loss_total1.backward()                           # 对主模型进行优化
    optimizer1.step()

    return rec_A_loss1,rec_A_loss2,pre_loss1,pre_loss2,loss_clu,clf_loss,loss_total1

def work2(model,netClf, X1, adj1,adjn1,X2, adj2,adjn2,optimizer2,sigma1,sigma2): 
    model.train()
    netClf.eval()
    loss_values,_,_,L_pred,G_pred,rec1,rec2,rec1_2,rec2_1,_= model(X1, adjn1, X2, adjn2,sigma1,sigma2)
    
   
    pre_loss11 = Pretrain_Loss(X1,rec2_1)    # 计算跨模态预测的矩阵损失
    pre_loss22 = Pretrain_Loss(X2,rec1_2)

   
    loss_con = loss_values.get("contrastiveloss")          # 从loss_values中提取对比模块的损失
    loss_total2=torch.mean( 10*pre_loss11  +  10*pre_loss22+0.05*loss_con)
    optimizer2.zero_grad()
    loss_total2.backward()
    optimizer2.step()

    return pre_loss11 ,pre_loss22,loss_con,loss_total2





def seed_everything(seed=317):          # 设计随机数
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_file", default=f"../dataset1/10x_PBMC/", type=str,help='9631')       # 模型参数的设置
    parser.add_argument('--seed', type=int, default='3407', help='seed')
    parser.add_argument("--highly_genes", default="2000", type=int)
    parser.add_argument("--lr", default="5e-5", type=float)
    parser.add_argument("--lr1", default="1e-5", type=float)
    parser.add_argument("--epochs1", default="600", type=int)
    parser.add_argument("--epochs2", default="660", type=int)
    parser.add_argument("--sigma", default="0.1", type=float)
    parser.add_argument("--n1", default="64", type=int)
    parser.add_argument("--n2", default="128", type=int)
    parser.add_argument("--n3", default="1024", type=int)
    parser.add_argument("--n4", default="10000", type=int)
    parser.add_argument("--n5", default="10000", type=int)
    parser.add_argument("--Nsample", default="10000", type=int)
    parser.add_argument('--save_dir', default='atac_pbmc10k2')
    parser.add_argument("--model_file", default="AE_weights_1.pth.tar", type=str)
    parser.add_argument("--n_clusters", default="19", type=int)
    parser.add_argument("--device",default="cuda:1", type=str)
    parser.add_argument("--batch", default="2000", type=int)
    parser.add_argument("--sigma1", default="1.5", type=float)
    parser.add_argument("--sigma2", default="1.5", type=float)

    parser.add_argument("--negative_samples_ratio",default=-1,type=float)
    parser.add_argument("--contrastive_similarity",default="cos",type=str,help="cos or gauss")
    parser.add_argument("--tau",default=0.1,type=float)
    parser.add_argument("--delta",default=0.01,type=float)
    parser.add_argument("--rel_sigma",default=0.1,type=float)
    parser.add_argument("--adaptive_contrastive_weight",default=True,type=bool)
    parser.add_argument("--funcs",default="contrastiveloss|SelfEntropy|ddcs", type=str)
    parser.add_argument("--weights",default=None, type=list)

    parser.add_argument("--Fusion_method",default="weightedfeature_mean", type=str)
    parser.add_argument("--n_views", default=2, type=int,help="mean or weighted_mean")

    parser.add_argument("--projector_config", default=False, type=bool)

    parser.add_argument("--ddc_hidden", default=32, type=int)
    parser.add_argument("--ddcuse_bn", default=False, type=bool)
    parser.add_argument("--ddc_direct", default=True, type=bool)
    parser.add_argument("--latent_projector_config", default=False, type=bool)




    args = parser.parse_args()
    seed_everything(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    device = args.device
    
    print(torch.__version__)
  
    x1 = ad.read_h5ad(args.data_file+"10x-Multiome-Pbmc10k-RNA.h5ad")        # 读取数据集
    x2 = ad.read_h5ad(args.data_file+"10x-Multiome-Pbmc10k-ATAC.h5ad")

    y= x1.obs['cell_type']
    y = y.tolist()
    print('----------shape of data------------------')           # 打印数据形状
    print(x1.shape)
    print(x2.shape)
    # print(y)
    y = encoder.fit_transform(y)       # 对字符串的label的转码到数字的label
    label_dict = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))        # 将这种转化关系进行记录，从而在可视化的时候进行对应
    file_path1 = os.path.join(args.save_dir, 'label_dict.pkl')
    with open(file_path1, 'wb') as f:
        pickle.dump(label_dict, f)
    print("字典保存完毕")
    print(len(y))

    x1_dense = x1.X
    x1.X = x1_dense.toarray()
    x2_dense = x2.X
    x2.X = x2_dense.toarray()
   
   
    adata1 = sc.AnnData(x1)
    adata1.obs['Group'] = y
                                              
    adata1 = normalize2(adata1, filter_min_counts=True, highly_genes=args.highly_genes,          # 对数据的预处理
                       size_factors=True, normalize_input=False,
                       logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y
    
    adata2 = normalize2(adata2, filter_min_counts=True, highly_genes=args.highly_genes,
                       size_factors=True, normalize_input=False,
                       logtrans_input=True)
    print('------------------------------------')


    print(adata1.shape)
    print(adata1)
    print(adata2.shape)
    print(adata2)

    
    Nsample1, Nfeature1 = np.shape(adata1.X)                   
    args.n4 = Nfeature1
    args.Nsample = Nsample1                      #根据过滤之后的数据对神经网络的第一层进行记录
    Nsample2, Nfeature2 = np.shape(adata2.X)
    args.n5 = Nfeature2




    args.n_clusters = len(np.unique(y))               
    print('-------len(y)----------------')
    print(len(np.unique(y)))

    adj1= get_adj(adata1.X)                # 构建细胞的拓扑结构图
    adj_n1 = GraphConv.preprocess(adj1)       # 对图进行归一化
    X1 = torch.from_numpy(adata1.X).to(device)
    adj1 = torch.from_numpy(adj1).to(device)
    adj_n1 = torch.from_numpy(adj_n1).to(device)

   
    adj2= get_adj(adata2.X)
    adj_n2 = GraphConv.preprocess(adj2)
    X2 = torch.from_numpy(adata2.X).to(device)
    adj2 = torch.from_numpy(adj2).to(device)
    adj_n2 = torch.from_numpy(adj_n2).to(device)
   
    print(args)                               
    netClf=FC_Classifier(nz=args.n1,n_out=1)    # 判别器的初始化
    model = Mymodel(args).to(device)
    model = model.float()
    pre_train(model,netClf,args,X1,y,adj1,adj_n1,X2,y,adj2,adj_n2,device)      #预训练
    alt_train(model,netClf,args,X1,y,adj1,adj_n1,X2,y,adj2,adj_n2,Nsample1,device)    #完整训练

    print("---------------------train finished-----------------")