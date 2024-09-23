import numpy as np
import scanpy as sc
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy import sparse as sp
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import torch
from munkres import Munkres


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

def clustering(args, z, y, adjn1):                                                          # 对每个模态得到编码和PPMI图进行评估，分别使用k-means和谱聚类
  labels_k=KMeans(n_clusters=args.n_clusters, n_init=20).fit_predict(z.data.cpu().numpy())
  labels_s = SpectralClustering(n_clusters=args.n_clusters,affinity="precomputed", assign_labels="discretize", n_init=20).fit_predict(adjn1.data.cpu().numpy())
  nmi_k ,ari_k= eva(y,labels_k)
  nmi_s ,ari_s= eva(y,labels_s)
  print('------kmeans---')
  print(nmi_k ,ari_k)
  print('------spectral-')
  print(nmi_s ,ari_s)
  labels = labels_s if (np.round(metrics.normalized_mutual_info_score(y, labels_s), 5)>=np.round(metrics.normalized_mutual_info_score(y, labels_k), 5)
  and np.round(metrics.adjusted_rand_score(y, labels_s), 5)>=np.round(metrics.adjusted_rand_score(y, labels_k), 5)) else labels_k
  nmi, ari = eva(y, labels)
  centers=computeCentroids(z.data.cpu().numpy(), labels)
  return nmi, ari, centers

def eva(y_true, y_pred):                              # 计算NMI和AIR
  nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
  ari = ari_score(y_true, y_pred)
  return nmi, ari

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10



def inference(network_model,X1, adj1, X2, adj2,y,num_views=2):

    network_model.eval()
    # mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    soft_vector = []
    pred_vectors = []
    labels_vector = []
    for v in range(num_views):
        pred_vectors.append([])
        
    with torch.no_grad():
        _,_, _, _, _, _, _, _, _,_,lbps = network_model(X1, adj1, X2, adj2)
        lbp = sum(lbps)/num_views
    for idx in range(num_views):
        pred_label = torch.argmax(lbps[idx], dim=1)
        pred_vectors[idx].extend(pred_label.detach().cpu().numpy())

    soft_vector.extend(lbp.detach().cpu().numpy())
    labels_vector.extend(y)
    for idx in range(num_views):
        pred_vectors[idx] = np.array(pred_vectors[idx])

    actual_num_samples = len(soft_vector)
    labels_vector = np.array(labels_vector).reshape(actual_num_samples)
    total_pred = np.argmax(np.array(soft_vector), axis=1)


    return total_pred, pred_vectors, labels_vector


def valid(network_model,X1, adj1, X2, adj2,y,num_views=2):

    total_pred, pred_vectors, labels_vector = inference(network_model, X1, adj1, X2, adj2,y)
    #num_views = len(mv_data.modalities)

    print("Clustering results on cluster assignments of each view:")
    for idx in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
        print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
                                                                                 idx+1, nmi,
                                                                                 idx+1, pur,
                                                                                 idx+1, ari))

    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari

def calculate_metrics(label, pred):
    acc = calculate_acc(label, pred)
    # nmi = v_measure_score(label, pred)
    nmi = nmi_score(label, pred)
    pur = calculate_purity(label, pred)
    ari = ari_score(label, pred)

    return acc, nmi, pur, ari


def calculate_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_sum_assignment(w.max() - w)

    # u = linear_sum_assignment(w.max() - w)
    # ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def calculate_purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster_index in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    return accuracy_score(y_true, y_voted_labels)

def ordered_cmat(labels, pred):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)   # 混淆矩阵
    ri, ci = linear_sum_assignment(-cmat)       # 计算聚类准确率
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered


def npy(t,to_cpu=False):
    """
    Convert a tensor to a numpy array.

    :param t: Input tensor
    :type t: th.Tensor
    :param to_cpu: Call the .cpu() method on `t`?
    :type to_cpu: bool
    :return: Numpy array
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}              # 转化为数组
    # Assuming t is a tensor.
    if to_cpu:
        return t.cpu().detach().numpy()
    return t.detach().numpy()


def calc_metrics(labels, pred):            # 计算聚类相关指标
    """
    Compute metrics.

    :param labels: Label tensor
    :type labels: th.Tensor
    :param pred: Predictions tensor
    :type pred: th.Tensor
    :return: Dictionary containing calculated metrics
    :rtype: dict
    """
    acc, cmat = ordered_cmat(labels, pred)
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": nmi_score(labels, pred, average_method="geometric"),
        "ari": ari_score(labels, pred)
    }
    return metrics


    
def eva_model(model,netClf, X1,adjn1,X2,adjn2,labels,Nsample,device):     # 评估模型训练能力
    predictions = []
    model.eval()
    netClf.eval()
    with torch.no_grad():
        _,_,_,_,_,_,_,_,_,pred = model.test(X1, adjn1, X2, adjn2)        # 对细胞的聚类结果进行分析
    pred = pred.cpu()  # 将张量从 CUDA 设备复制到主机内存
    predictions.append(npy(pred).argmax(axis=1))
    #labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    res=calc_metrics(labels,predictions)
    return res

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def save_result(model,netClf, X1,adjn1,X2,adjn2,labels,Nsample,device,save_dir): #保存相关文件信息
    predictions = []
    model.eval()
    netClf.eval()
    with torch.no_grad():
        _,z_L,z_G,_,_,_,_,rec1_2,rec2_1,final_Z,pred = model.save(X1, adjn1, X2, adjn2)

    final_latent1 = z_L.cpu().numpy()
    np.savetxt(save_dir + "/"  + "embeddingrna.csv", final_latent1, delimiter=",")

    final_latent2 = z_G.cpu().numpy()
    np.savetxt(save_dir + "/"  + "embeddingatac.csv", final_latent2, delimiter=",")

    final_latentall = final_Z.cpu().numpy()
    np.savetxt(save_dir + "/"  + "embeddingall.csv", final_latentall, delimiter=",")

    recon1_2 = rec1_2.cpu().numpy()
    np.savetxt(save_dir + "/"  + "rec1_2.csv", recon1_2, delimiter=",")

    recon2_1 = rec2_1.cpu().numpy()
    np.savetxt(save_dir + "/"  + "rec2_1.csv", recon2_1, delimiter=",")

    pred = pred.cpu()  # 将张量从 CUDA 设备复制到主机内存
    predictions.append(npy(pred).argmax(axis=1))
    #labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    y_pred_ = best_map(labels, predictions)                        # 将聚类结果的label与初始的编码label进行最佳对应
    np.savetxt(save_dir + "/"  + "predicted.csv", y_pred_, delimiter=",")
    

