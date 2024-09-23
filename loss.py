import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.distributions import Normal, kl_divergence as kl
import torch as th
import torch.nn as nn

# A_rec_loss=tf.reduce_mean(MSE(self.adj, A_out))
def mse_loss(y_true, y_pred):

    mask = torch.sign(y_true)

    y_pred = y_pred.float()
    y_true = y_true.float()

    ret = torch.pow( (y_pred - y_true) * mask , 2)
    return torch.sum( ret, dim = 1 )


def A_recloss(adj, A_out):
    A_rec_loss=torch.mean(mse_loss(adj, A_out))
    return A_rec_loss


import kernel

EPSILON = 1E-9
DEBUG_MODE = False


def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms(Not adopted)
# ======================================================================================================================

class LossTerm:
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net, args, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __call__(self, net, args, extra):
        return d_cs(net.output, extra["hidden_kernel"], args.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def __call__(self, net, cfg, extra):
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


class DDC2Flipped(LossTerm):
    """
    Flipped version of the L_2 loss from DDC. Used by EAMC
    """

    def __call__(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __init__(self, args):
        super().__init__()
        self.eye = th.eye(args.n_clusters, device=args.device)

    def __call__(self, net, args, extra):
        m = th.exp(-kernel.cdist(net.output, self.eye))
        return d_cs(m, extra["hidden_kernel"], args.n_clusters)




class ContrastiveLoss(LossTerm):
    """\
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """
    def __init__(self,args):
        super().__init__()
        self.n_output = args.n_clusters
        self.eye = torch.eye(args.n_clusters, device=args.device)
        self.sampling_ratio = 0.25
        self.tau = 0.1


    @staticmethod
    def _cosine_similarity(projections):
        h = F.normalize(projections, p=2, dim=1)
        return h @ h.t()

    def _draw_negative_samples(self, predictions, v, pos_indices):
        predictions = torch.cat(v * [predictions], dim=0)
        weights = (1 - self.eye[predictions])[:, predictions[[pos_indices]]].T
        n_negative_samples = int(self.sampling_ratio * predictions.size(0))
        negative_sample_indices = torch.multinomial(
            weights, n_negative_samples, replacement=True
        )
        return negative_sample_indices

    @staticmethod
    def _get_positive_samples(logits, v, n):
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = torch.diagonal(logits, offset=diagonal_offset)
            _lower = torch.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = torch.arange(0, diag_length)
            _lower_inds = torch.arange(i * n, v * n)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = torch.cat(diagonals, dim=0)
        pos_inds = torch.cat(inds, dim=0)
        return pos, pos_inds

    def __call__(self, net, args, extra):
        # if model.n_modality == 1:
        #     return 0, [0] * model.n_head

        n_sample = args.Nsample

        # total_loss = 0
        # total_loss = torch.tensor(0, device=args.device, dtype=torch.float)
        # head_losses = None

        logits = (
                ContrastiveLoss._cosine_similarity(net.latent_projection)
                / self.tau
        )
        pos, pos_inds = ContrastiveLoss._get_positive_samples(
            logits, 2, n_sample
        )

        predictions = torch.argmax(net.output,axis =1)
        if len(torch.unique(predictions)) > 1:
            neg_inds = self._draw_negative_samples(
                predictions, 2, pos_inds
            )
            neg = logits[pos_inds.view(-1, 1), neg_inds]
            inputs = torch.cat((pos.view(-1, 1), neg), dim=1)
            labels = torch.zeros(
               2 * (2 - 1) * n_sample,
                device=args.device,
                dtype=torch.long,
            )
            loss = F.cross_entropy(inputs, labels)
    

        else:
            loss = 0.0
        

    

        return loss



class Contrastive(LossTerm):
    large_num = 1e9

    def __init__(self, args):
        """
        Contrastive loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        # Select which implementation to use
        if args.negative_samples_ratio == -1:
            self._loss_func = self._loss_without_negative_sampling
        else:
            self.eye = th.eye(args.n_clusters, device=args.device)
            self._loss_func = self._loss_with_negative_sampling

        # Set similarity function
        if args.contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif args.contrastive_similarity == "gauss":
            self.similarity_func = kernel.vector_kernel
        else:
            raise RuntimeError(f"Invalid contrastive similarity: {args.contrastive_similarity}")

    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @staticmethod
    def get_weight(net):
        w = th.min(th.nn.functional.softmax(net.fusion.weights.detach(), dim=0))
        return w

    @classmethod
    def _normalized_projections(cls, net):
        n = net.projections.size(0) // 2
        h1, h2 = net.projections[:n], net.projections[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()

    def _draw_negative_samples(self, net, args, v, pos_indices):
        """
        Construct set of negative samples.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param v: Number of views
        :type v: int
        :param pos_indices: Row indices of the positive samples in the concatenated similarity matrix
        :type pos_indices: th.Tensor
        :return: Indices of negative samples
        :rtype: th.Tensor
        """
        cat = net.output.detach().argmax(dim=1)
        cat = th.cat(v * [cat], dim=0)

        weights = (1 - self.eye[cat])[:, cat[[pos_indices]]].T
        n_negative_samples = int(args.negative_samples_ratio * cat.size(0))
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=True)
        if DEBUG_MODE:
            self._check_negative_samples_valid(cat, pos_indices, negative_sample_indices)
        return negative_sample_indices

    @staticmethod
    def _check_negative_samples_valid(cat, pos_indices, neg_indices):
        pos_cats = cat[pos_indices].view(-1, 1)
        neg_cats = cat[neg_indices]
        assert (pos_cats != neg_cats).detach().cpu().numpy().all()

    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        Get positive samples

        :param logits: Input similarities
        :type logits: th.Tensor
        :param v: Number of views
        :type v: int
        :param n: Number of samples per view (batch size)
        :type n: int
        :return: Similarities of positive pairs, and their indices
        :rtype: Tuple[th.Tensor, th.Tensor]
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length)
            _lower_inds = th.arange(i * n, v * n)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, net, args, extra):
        """
        Contrastive loss implementation with negative sampling.

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param extra:
        :type extra:
        :return: Loss value
        :rtype: th.Tensor
        """
        n = net.output.size(0)
        v = len(net.backbone_outputs)
        logits = self.similarity_func(net.projections) / args.tau

        pos, pos_inds = self._get_positive_samples(logits, v, n)
        neg_inds = self._draw_negative_samples(net, args, v, pos_inds)
        neg = logits[pos_inds.view(-1, 1), neg_inds]

        inputs = th.cat((pos.view(-1, 1), neg), dim=1)
        labels = th.zeros(v * (v - 1) * n, device=args.device, dtype=th.long)
        loss = th.nn.functional.cross_entropy(inputs, labels)

        if args.adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return args.delta * loss

    def _loss_without_negative_sampling(self, net, args, extra):
        """
        Contrastive loss implementation without negative sampling.
        Adapted from: https://github.com/google-research/simclr/blob/master/objective.py

        :param net: Model
        :type net: Union[models.simple_mvc.SiMVC, models.contrastive_mvc.CoMVC]
        :param cfg: Loss config
        :type cfg: config.defaults.Loss
        :param extra:
        :type extra:
        :return:
        :rtype:
        """
        assert len(net.backbone_outputs) == 2, "Contrastive loss without negative sampling only supports 2 views."
        n, h1, h2 = self._normalized_projections(net)

        labels = th.arange(0, n, device=args.device, dtype=th.long)
        masks = th.eye(n, device=args.device)

        logits_aa = ((h1 @ h1.t()) / args.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / args.tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / args.tau
        logits_ba = (h2 @ h1.t()) / args.tau

        loss_a = th.nn.functional.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = th.nn.functional.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)

        if args.adaptive_contrastive_weight:
            loss *= self.get_weight(net)

        return args.delta * loss

    def __call__(self, net, args, extra):
        return self._loss_func(net, args, extra)


# ======================================================================================================================
# Loss Term
# ======================================================================================================================





def hidden_kernel(net, args):
    return kernel.vector_kernel(net.hidden, args.rel_sigma)



class SelfEntropyLoss( LossTerm):
  
    """
    Entropy regularization to prevent trivial solution.
    """

    def __init__(self,args):
        super().__init__()
       

    def __call__(self, net, args, extra):
        eps = 1e-8
        cluster_outputs = net.output
        prob_mean = cluster_outputs.mean(dim=0)
        prob_mean[(prob_mean < eps).data] = eps
        loss = (prob_mean * torch.log(prob_mean)).sum()


        return loss

class ContrastiveLoss(LossTerm):
    """\
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """
    def __init__(self,args):
        super().__init__()
        self.n_output = args.n_clusters
        self.eye = torch.eye(args.n_clusters, device=args.device)
        self.sampling_ratio = 0.25
        self.tau = 0.1


    @staticmethod
    def _cosine_similarity(projections):
        h = F.normalize(projections, p=2, dim=1)
        return h @ h.t()

    def _draw_negative_samples(self, predictions, v, pos_indices):
        predictions = torch.cat(v * [predictions], dim=0)
        weights = (1 - self.eye[predictions])[:, predictions[[pos_indices]]].T
        n_negative_samples = int(self.sampling_ratio * predictions.size(0))
        negative_sample_indices = torch.multinomial(
            weights, n_negative_samples, replacement=True
        )
        return negative_sample_indices

    @staticmethod
    def _get_positive_samples(logits, v, n):
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = torch.diagonal(logits, offset=diagonal_offset)
            _lower = torch.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = torch.arange(0, diag_length)
            _lower_inds = torch.arange(i * n, v * n)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = torch.cat(diagonals, dim=0)
        pos_inds = torch.cat(inds, dim=0)
        return pos, pos_inds

    def __call__(self, net, args, extra):
        # if model.n_modality == 1:
        #     return 0, [0] * model.n_head

        n_sample = args.Nsample

        # total_loss = 0
        # total_loss = torch.tensor(0, device=args.device, dtype=torch.float)
        # head_losses = None

        logits = (
                ContrastiveLoss._cosine_similarity(net.latent_projection)
                / self.tau
        )
        pos, pos_inds = ContrastiveLoss._get_positive_samples(
            logits, 2, n_sample
        )

        predictions = torch.argmax(net.output,axis =1)
        if len(torch.unique(predictions)) > 1:
            neg_inds = self._draw_negative_samples(
                predictions, 2, pos_inds
            )
            neg = logits[pos_inds.view(-1, 1), neg_inds]
            inputs = torch.cat((pos.view(-1, 1), neg), dim=1)
            labels = torch.zeros(
               2 * (2 - 1) * n_sample,
                device=args.device,
                dtype=torch.long,
            )
            loss = F.cross_entropy(inputs, labels)
    

        else:
            loss = 0.0
        

    

        return loss

    
class DDCLoss( LossTerm):
    """
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """

    def __init__(self,args):
        super().__init__()
        self.n_output = args.n_clusters
        self.eye = torch.eye(args.n_clusters, device=args.device)

    @staticmethod
    def triu(X):
        """\ 
        Sum of strictly upper triangular part.
        """
        return torch.sum(torch.triu(X, diagonal=1))

    @staticmethod
    def _atleast_epsilon(X, eps=1e-9):
        """
        Ensure that all elements are >= `eps`.
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    @staticmethod
    def d_cs(A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(
            torch.diagonal(nom), 0
        )

        nom = DDCLoss._atleast_epsilon(nom)
        dnom_squared = DDCLoss._atleast_epsilon(dnom_squared, eps=EPSILON** 2)

        d = (
            2
            / (n_clusters * (n_clusters - 1))
            * DDCLoss.triu(nom / torch.sqrt(dnom_squared))
        )
        return d

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON ):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k


    @staticmethod
    def cdist(X, Y):
        """\
        Pairwise distance between rows of X and rows of Y.
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    @staticmethod
    def vector_kernel(x, rel_sigma=0.15):
        """\
        Compute a kernel matrix from the rows of a matrix.
        """
        return DDCLoss.kernel_from_distance_matrix(DDCLoss.cdist(x, x), rel_sigma)

    def __call__(self, net, args, extra):

        hidden = net.hidden
        cluster_outputs = net.output
        hidden_kernel = DDCLoss.vector_kernel(hidden)
        # L_1 loss
        loss = DDCLoss.d_cs(cluster_outputs, hidden_kernel, self.n_output)

        # L_3 loss
        m = torch.exp(-DDCLoss.cdist(cluster_outputs, self.eye))
        loss += DDCLoss.d_cs(m, hidden_kernel, self.n_output)
        
        return loss

# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(nn.Module):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_3": DDC3,
        "contrast": Contrastive,
        "SelfEntropy":SelfEntropyLoss,       # 聚类损失的第一项
        "ddcs":DDCLoss,             # 聚类损失的第二和第三项
        "contrastiveloss":ContrastiveLoss  # 对比损失
        
    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, args):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.args = args

        self.names = args.funcs.split("|")
        self.weights = args.weights if args.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            self.terms.append(self.TERM_CLASSES[term_name](args))          

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.args) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.args, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])     # 计算各项的损失和总体损失
        return loss_values
