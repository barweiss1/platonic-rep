import torch
import torchaudio.functional as TAF

import numpy as np
from sklearn.cross_decomposition import CCA

try:
    import pymp
    pymp_available = True
except ImportError:
    pymp_available = False
    print("Please install the pymp library using `pip install pymp` to speed up non-batched metrics")


class AlignmentMetrics:

    SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "mutual_knn_dist",
        "lcs_knn",
        "cka",
        "cka_rbf",
        "unbiased_cka",
        "cka_rbf_quantile",
        "cknna",
        "svcca",
        "edit_distance_knn",
        "nn_rwka",
        "ip_nn_rwka",
        "asym_nn_rwka",
        "softmax_rwka",
        "softmax_cka",
        "rbf_rwka",
        "rbf_rwka_quantile",
    ]
    
    # Define which parameter each metric sweeps over and its range
    SWEEP_PARAMS = {
        "cycle_knn": {"param": "topk", "min": 3, "max": 500},
        "mutual_knn": {"param": "topk", "min": 3, "max": 500},
        "mutual_knn_dist": {"param": "topk", "min": 3, "max": 500},
        "lcs_knn": {"param": "topk", "min": 3, "max": 500},
        "cknna": {"param": "topk", "min": 3, "max": 500},
        "edit_distance_knn": {"param": "topk", "min": 5, "max": 500},
        "cka_rbf": {"param": "rbf_sigma", "min": 0.01, "max": 5.0},
        "cka_rbf_quantile": {"param": "quantile", "min": 0.01, "max": 0.7},
        "nn_rwka": {"param": "topk", "min": 3, "max": 500},
        "ip_nn_rwka": {"param": "topk", "min": 3, "max": 500},
        "asym_nn_rwka": {"param": "topk", "min": 3, "max": 500},
        "softmax_rwka": {"param": "temperature", "min": 1e-5, "max": 10.0},
        "softmax_cka": {"param": "temperature", "min": 1e-5, "max": 10.0},        
        "cka": {"param": None, "min": None, "max": None},  # No sweep
        "unbiased_cka": {"param": None, "min": None, "max": None},  # No sweep
        "svcca": {"param": None, "min": None, "max": None},  # No sweep
        "rbf_rwka": {"param": "rbf_sigma", "min": 0.01, "max": 5.0},  # RBF kernel bandwidth,
        "rbf_rwka_quantile": {"param": "quantile", "min": 0.01, "max": 0.7},
    }

    def _to_torch_tensor(feats):
        if torch.is_tensor(feats):
            return feats
        return torch.from_numpy(np.asarray(feats))

    @staticmethod
    def measure(metric, feats_A, feats_B, *args, **kwargs):
        """ metric is a string for the function """

        if metric not in AlignmentMetrics.SUPPORTED_METRICS:
            raise ValueError(f"Unrecognized metric: {metric}")
        
        feats_A = AlignmentMetrics._to_torch_tensor(feats_A)
        feats_B = AlignmentMetrics._to_torch_tensor(feats_B)

        # Map cka_rbf to cka with rbf kernel
        if metric == "cka_rbf":
            kwargs['kernel_metric'] = 'rbf'
            return AlignmentMetrics.cka(feats_A, feats_B, *args, **kwargs)
        elif metric == "cka_rbf_quantile":
            kwargs['kernel_metric'] = 'rbf'
            return AlignmentMetrics.cka(feats_A, feats_B, *args, **kwargs)
        elif metric == "nn_rwka":
            kwargs['use_distance'] = True
            kwargs['symmetric'] = True
            return AlignmentMetrics.nn_rwka(feats_A, feats_B, *args, **kwargs)
        elif metric == "ip_nn_rwka":
            kwargs['use_distance'] = False
            kwargs['symmetric'] = True
            return AlignmentMetrics.nn_rwka(feats_A, feats_B, *args, **kwargs)
        elif metric == "asym_nn_rwka":
            kwargs['use_distance'] = True
            kwargs['symmetric'] = False
            return AlignmentMetrics.nn_rwka(feats_A, feats_B, *args, **kwargs)
        elif metric == "rbf_rwka_quantile":
            return AlignmentMetrics.rbf_rwka(feats_A, feats_B, *args, **kwargs)
        elif metric == "mutual_knn_dist":
            kwargs['use_distance'] = True
            return AlignmentMetrics.mutual_knn(feats_A, feats_B, *args, **kwargs)
        
        return getattr(AlignmentMetrics, metric)(feats_A, feats_B, *args, **kwargs)


    @staticmethod
    def cycle_knn(feats_A, feats_B, topk):
        """
        LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            acc: a float representing the accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   
        return compute_knn_accuracy(knn_A[knn_B]).item()


    @staticmethod
    def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()
    
    
    @staticmethod
    def lcs_knn(feats_A, feats_B, topk):
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)        
        score = longest_ordinal_sequence(knn_A, knn_B).float().mean()
        return score
    
    
    @staticmethod
    @staticmethod
    def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False, median=True, quantile=None):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""
        
        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            K = compute_rbf_kernel(feats_A, rbf_sigma=rbf_sigma, median=median, quantile=quantile)
            L = compute_rbf_kernel(feats_B, rbf_sigma=rbf_sigma, median=median, quantile=quantile)

        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        # print(f'rbf_sigma: {rbf_sigma}, hsic_kl: {hsic_kl.item()}, hsic_kk: {hsic_kk.item()}, hsic_ll: {hsic_ll.item()}')
        denominator = torch.sqrt(hsic_kk * hsic_ll).item()
        denominator = check_division_by_zero_warning("CKA", denominator)
        cka_value = hsic_kl / denominator        
        return cka_value.item()
    
    # ---------------------- My Similarity Metrics ----------------------
    @staticmethod
    def softmax_rwka(feats_A, feats_B, temperature, unbiased=False, range_based=False):
        """
        Computes the softmax-based CKA between features.
        The inner products are converted to similarity matrices using softmax with a temperature parameter.
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim
            temperature: A float representing the temperature for softmax
        Returns:
            A float representing the softmax-based CKA similarity
        """

        K = compute_softmax_kernel(feats_A, temperature, range_based=range_based)
        L = compute_softmax_kernel(feats_B, temperature, range_based=range_based)

        # sim_kl = rw_similarity(K, L, unbiased)
        # sim_kk = rw_similarity(K, K, unbiased)
        # sim_ll = rw_similarity(L, L, unbiased)

        # since K and L are markovian matrices their inner product is defines a random walk similarity
        sim_kl = torch.sum(K * L.T)
        sim_kk = torch.sum(K * K.T)
        sim_ll = torch.sum(L * L.T)
                
        denominator = torch.sqrt(sim_kk * sim_ll).item()
        denominator = check_division_by_zero_warning("Softmax RWKA", denominator)

        softmax_score = sim_kl.item() / denominator
        print(f"Temperature: {temperature}, sim_kl: {sim_kl.item()}, sim_kk: {sim_kk.item()}, sim_ll: {sim_ll.item()}, softmax_score: {softmax_score}")
        if torch.isnan(torch.tensor(softmax_score)):
            print(f"Warning: Softmax RWKA returned NaN. sim_kl: {sim_kl.item()}, sim_kk: {sim_kk.item()}, sim_ll: {sim_ll.item()}, temperature: {temperature}")
            return 0.0  
        return softmax_score
    

    @staticmethod
    def softmax_cka(feats_A, feats_B, temperature, unbiased=False):
        """
        Computes the softmax-based CKA between features.
        The inner products are converted to similarity matrices using softmax with a temperature parameter.
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim
            temperature: A float representing the temperature for softmax
        Returns:
            A float representing the softmax-based CKA similarity
        """
        K = compute_softmax_kernel(feats=feats_A, temperature=temperature, median=True)
        L = compute_softmax_kernel(feats=feats_B, temperature=temperature, median=True)

        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        sim_kl = hsic_fn(K, L)
        sim_kk = hsic_fn(K, K)
        sim_ll = hsic_fn(L, L)
                
        return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-9).item()
    


    @staticmethod
    def rbf_rwka(feats_A, feats_B, rbf_sigma, unbiased=False, median=True):
        """
        Computes the diffusion-based CKA between features.
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim
            rbf_sigma: A float representing the bandwidth scaling factor for the RBF kernel
        """
        K = compute_rbf_kernel(feats_A, rbf_sigma=rbf_sigma, median=median)
        L = compute_rbf_kernel(feats_B, rbf_sigma=rbf_sigma, median=median)

        sim_kl = rw_similarity(K, L, unbiased, symmetric=False)
        sim_kk = rw_similarity(K, K, unbiased, symmetric=False)
        sim_ll = rw_similarity(L, L, unbiased, symmetric=False)

        denominator = check_division_by_zero_warning("RBF RWKA", torch.sqrt(sim_kk * sim_ll))
        return sim_kl.item() / denominator

    @staticmethod
    def nn_rwka(
        feats_A, 
        feats_B, 
        topk, 
        unbiased=False,
        use_distance=True,
        symmetric=False):
        """
        Computes the RWKA on the k-NN graph between features.
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim
            topk: The number of nearest neighbors to consider in the random walk
        Returns:    
            A float representing the NN-RWKA similarity
        """
        n = feats_A.shape[0]
                
        if topk < 2:
            raise ValueError("NN-RWKA requires topk >= 2")
        
        if topk is None:
            topk = feats_A.shape[0] - 1
        
        K = compute_nearest_neighbors_graph(feats_A, topk, 
                                            use_distance=use_distance, 
                                            symmetric=symmetric)
        L = compute_nearest_neighbors_graph(feats_B, topk, 
                                            use_distance=use_distance, 
                                            symmetric=symmetric)


        sim_kl = rw_similarity(K, L, unbiased, symmetric=False)
        sim_kk = rw_similarity(K, K, unbiased, symmetric=False)
        sim_ll = rw_similarity(L, L, unbiased, symmetric=False)

        denominator = check_division_by_zero_warning("NN-RWKA", torch.sqrt(sim_kk * sim_ll))
        return sim_kl.item() / denominator

    @staticmethod
    def unbiased_cka(*args, **kwargs):
        kwargs['unbiased'] = True
        return AlignmentMetrics.cka(*args, **kwargs)
    
    
    @staticmethod
    def svcca(feats_A, feats_B, cca_dim=10):

        # Center and scale the activations
        def preprocess_activations(act):
            act = act - torch.mean(act, axis=0)
            act = act / (torch.std(act, axis=0) + 1e-8)
            return act

        feats_A = preprocess_activations(feats_A)
        feats_B = preprocess_activations(feats_B)

        # Compute SVD
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
        
        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # sometimes it goes to nan, this is just to avoid that
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute SVCCA similarity
        svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
        )
        return svcca_similarity
    
    
    @staticmethod
    def edit_distance_knn(feats_A, feats_B, topk):
        """
        Computes the edit distance between the nearest neighbors of feats_A and feats_B.
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)
        
        # given N x topk with integer entries, compute edit distance
        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        edit_distance = compute_distance(knn_A, knn_B, TAF.edit_distance)
        return 1 - torch.mean(edit_distance) / topk
    
    
    @staticmethod
    def cknna(feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
        """ similarity only cka variant """
        n = feats_A.shape[0]
                
        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")
        
        if topk is None:
            topk = feats_A.shape[0] - 1
                            
        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        device = feats_A.device

        def similarity(K, L, topk):                         
            if unbiased:            
                K_hat = K.clone().fill_diagonal_(float("-inf"))
                L_hat = L.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = K, L

            # get topk indices for each row
            # if unbiased we cannot attend to the diagonal unless full topk
            # else we can attend to the diagonal
            _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
            
            # create masks for nearest neighbors
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
            
            # intersection of nearest neighbors
            mask = mask_K * mask_L
                        
            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = hsic_unbiased(mask * K, mask * L)
                else:
                    sim = hsic_biased(mask * K, mask * L)
            return sim

        sim_kl = similarity(K, L, topk)
        sim_kk = similarity(K, K, topk)
        sim_ll = similarity(L, L, topk)

        denominator = torch.sqrt(sim_kk * sim_ll).item()

        denominator = check_division_by_zero_warning("CKNNA", denominator)

        return sim_kl.item() / denominator

def check_division_by_zero_warning(metric_name, denominator):
    if denominator < 1e-10:
        print(f"Warning: {metric_name} denominator is very small ({denominator}), adding epsilon to avoid division by zero.")
    return denominator + 1e-10

def rw_similarity(K, L, unbiased=False, symmetric=True):
    """
    Computes the similarity between two kernels K and L for random-walk based methods.
    Used for random walk kernel alignment (RWKA) similarity.
    Args:
        K: Kernel matrix of shape (N, N)
        L: Kernel matrix of shape (N, N)
        unbiased: Whether to use the unbiased estimator
    """
    if unbiased:            
        K_hat = K.clone().fill_diagonal_(0.0)
        L_hat = L.clone().fill_diagonal_(0.0)
    else:
        K_hat, L_hat = K, L
    
    K_norm = normalize_random_walk_kernel(K_hat, symmetric=symmetric)
    L_norm = normalize_random_walk_kernel(L_hat, symmetric=symmetric)

    # Compute the similarity as the sum of element-wise product of the normalized kernels (mathematically equivalent to trace(K_norm @ L_norm))
    sim = torch.sum(K_norm * L_norm.T)
    
    return sim

def normalize_random_walk_kernel(K, symmetric=True):
    """ Compute the normalized kernel for random walk based methods """
    device = K.device
    col_sum = torch.sum(K, dim=0)
    col_sum[col_sum == 0] = 1.0  # avoid division by zero
    if symmetric:
        D_K_mhalf  = torch.diag(col_sum ** (-0.5)).to(device)
        return D_K_mhalf @ K @ D_K_mhalf
    else:
        D_K_inv = torch.diag(1.0 / col_sum).to(device)
        return K @ D_K_inv

def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

    
def compute_knn_accuracy(knn):
    """
    Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
    Args:
        knn: a torch tensor of shape N x topk
    Returns:
        acc: a float representing the accuracy
    """
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc
    

def compute_nearest_neighbors(feats, topk=1, use_distance=False):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
        use_distance: if True, use Euclidean distance instead of correlation/similarity
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    
    if use_distance:
        # Use negative squared distance for efficient computation
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        feat_norm_sq = (feats ** 2).sum(dim=1, keepdim=True)
        dist_sq = feat_norm_sq + feat_norm_sq.T - 2 * (feats @ feats.T)
        # Use negative distance so we can still use descending=False for closest neighbors
        knn = dist_sq.fill_diagonal_(1e8).argsort(dim=1, descending=False)[:, :topk]
    else:
        # Use correlation/similarity (original behavior)
        knn = (
            (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
        )
    return knn


def compute_nearest_neighbors_graph(
        feats, 
        topk=3, 
        use_distance=True,
        symmetric=True):
    """
    Compute the nearest neighbors graph of feats based on Euclidean distance
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
        use_distance: if True, use Euclidean distance instead of correlation/similarity
    Returns:
        knn_graph: a torch tensor of shape N x N, where knn_graph[i, j] = 1 if j is a nearest neighbor of i
    """
    n = feats.shape[0]
    knn = compute_nearest_neighbors(feats, topk, use_distance=use_distance)
    knn_graph = torch.zeros(n, n, device=feats.device)
    knn_graph.scatter_(1, knn, 1)
    if symmetric:
        knn_graph = torch.maximum(knn_graph, knn_graph.T)
    return knn_graph


def compute_softmax_kernel(feats, temperature=0.1, range_based=False):
    """
        Computes the softmax kernel from features, 
        with an option to use the median heuristic for temperature selection.
        The kernel is in double precision to avoid underflow issues in the exponential computation, 
        and in further processing (e.g. multiplication with another kernel).
        Args:
            feats: a torch tensor of shape N x D
            temperature: a float representing the temperature for the softmax kernel
            range_based: if True, use the range-based heuristic to set the temperature based on pairwise distances
        Returns:
            K: a torch tensor of shape N x N representing the softmax kernel matrix
    """
    # COMPUTES SOFTMAX KERNEL with higher precision to avoid numerical issues
    # Convert to float64 for stable computation
    feats_hp = feats.double()

    K = torch.mm(feats_hp, feats_hp.T)

    if range_based:
        range = K.max() - K.min()
        temperature_K = range * temperature
    else:
        # use median heuristic for temperature but first shift inner products to be positive to 
        # avoid negative temperatures, this is possible since softmax is shift invariant
        median = K.median() - K.min()
        temperature_K = median * temperature
    
    def stable_softmax(x, dim=-1):
        # Subtract the maximum value for numerical stability
        z = x - torch.max(x, dim=dim, keepdim=True)[0]
        exp_z = torch.exp(z)
        return exp_z / torch.sum(exp_z, dim=dim, keepdim=True)

    # Compute softmax kernel in float64 precision to avoid underflow issues in multiplication
    # K = stable_softmax(K / temperature_K, dim=1)
    K = torch.softmax(K / temperature_K, dim=1)

    return K


def compute_rbf_kernel(feats, rbf_sigma=1.0, median=True, quantile=None):
    """
        Computes the rbf kernel from features, 
        with an option to use the median heuristic for bandwidth selection.
        The kernel is in double precision to avoid underflow issues in the exponential computation, 
        and in further processing (e.g. multiplication with another kernel).
        Args:
            feats: a torch tensor of shape N x D
            rbf_sigma: a float representing the bandwidth scaling factor for the RBF kernel
            median: if True, use the median heuristic to set the bandwidth based on pairwise distances
            quantile: if not None, use the quantile heuristic to set the bandwidth based on pairwise distances
        Returns:
            K: a torch tensor of shape N x N representing the RBF kernel matrix
    """
    # COMPUTES RBF KERNEL with higher precision to avoid numerical issues
    # Convert to float64 for stable computation
    feats_hp = feats.double()

    K = torch.cdist(feats_hp, feats_hp)

    if quantile is not None:
        tril_indices = torch.tril_indices(K.shape[0], K.shape[1], offset=-1)
        rbf_sigma_K = torch.quantile(K[tril_indices[0], tril_indices[1]], q=float(quantile)).item()
    elif median:
        # use median heuristic for bandwidth using lower triangular part (excluding diagonal)
        tril_indices = torch.tril_indices(K.shape[0], K.shape[1], offset=-1)
        rbf_sigma_K = torch.median(K[tril_indices[0], tril_indices[1]]).item() * rbf_sigma
    else:
        rbf_sigma_K = rbf_sigma
    
    # Compute RBF kernel in float64 precision to avoid underflow issues in multiplication
    K = torch.exp(- K ** 2 / (2 * rbf_sigma_K ** 2))

    return K


def longest_ordinal_sequence(X, Y):
    """ For each pair in X and Y, compute the length of the longest sub-sequence (LCS) """
    
    def lcs_length(x, y):
        """
        Compute the length of the longest common subsequence between two sequences.
        This is a classic dynamic programming implementation.
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = compute_distance(X, Y, lcs_length)
    return lcs


def compute_distance(X, Y, dist_fn):
    """ compute distance in parallel"""
    B, N = X.shape
    distances = np.zeros(B)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()

    if pymp_available:
        with pymp.Parallel(4) as p:
            for i in p.range(B):
                distances[i] = dist_fn(X[i], Y[i])
    else:
        for i in range(B):
            distances[i] = dist_fn(X[i], Y[i])
    return torch.tensor(distances)


def remove_outliers(feats, q, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        # sorts the whole tensor and gets the q-th percentile
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        # quantile for element in the tensor and take the average
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)



if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    feats_A = torch.randn(64, 8192)
    feats_B = torch.randn(64, 8192)
    feats_A = F.normalize(feats_A, dim=-1)
    feats_B = F.normalize(feats_B, dim=-1)

    import time 
    trials = 10

    t0 = time.time()
    for metric in AlignmentMetrics.SUPPORTED_METRICS:

        scores, times = [], []
        for t in range(trials):
            t_st = time.time()

            kwargs = {}
            if 'nn' in metric:
                kwargs['topk'] = 10
            if 'cca' in metric:
                kwargs['cca_dim'] = 10
            if 'kernel' in metric:
                kwargs['dist'] = 'sample'

            score = AlignmentMetrics.measure(metric, feats_A, feats_B, **kwargs)
            scores.append(score)
            times.append(time.time() - t_st)        
        print(f"{metric.rjust(20)}: {np.mean(scores):1.3f} [elapsed: {np.mean(times):.2f}s]")

    print(f'Total time: {time.time() - t0:.2f}s')