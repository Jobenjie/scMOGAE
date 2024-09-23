
import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import h5py
import utils as utils

def dopca(X, dim):                            
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        adata.obs['size_factors'] = np.log(np.sum(adata.X, axis=1))
    else:
        adata.obs['size_factors'] = 1.0

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def normalize2(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):     # 数据预处理，过滤低质量的细胞和基因，挑选出高变基因
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


CHUNK_SIZE = 20000
import scipy
from scipy.sparse import issparse




def preprocessing_atac(
        adata, 
        min_genes=200, 
        min_cells=3, # 0.01, 
        n_top_genes=30000,
        target_sum=None,
        chunk_size=CHUNK_SIZE,
        log=None
    ):
    """
    preprocessing      SCALE方法中对atac的处理
    """
    print('Raw dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
        
    adata.X[adata.X>1] = 1
    
    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    if log: log.info('Filtering genes')
    if min_cells < 1:
        min_cells = min_cells * adata.shape[0]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if n_top_genes != -1:
        if log: log.info('Finding variable features')
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
        # adata = epi.pp.select_var_feature(adata, nb_features=n_top_genes, show=False, copy=True)
    
    # if log: log.infor('Normalizing total per cell')
    # sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if log: log.info('Batch specific maxabs scaling')
    
#     adata.X = maxabs_scale(adata.X)
#     adata = batch_scale(adata, chunk_size=chunk_size)
    
    print('Processed dataset shape: {}'.format(adata.shape))
    return adata
    