import numpy as np
from modAL import ActiveLearner
from pyhard import get_seed
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def mcm_sampling(learner: ActiveLearner, X: np.ndarray, n_instances: int = 1, pool_factor: int = 5) -> np.ndarray:
    """Estratégia MCM - Marginal Cluster Mean (Xu et al., 2003) compatível com modAL."""
    if len(X) <= n_instances:
        return np.arange(len(X))

    proba = learner.predict_proba(X)
    if proba.shape[1] < 2:
        rng = np.random.default_rng(get_seed())
        return rng.choice(len(X), size=n_instances, replace=False)

    part = np.partition(-proba, 1, axis=1)
    margin = -part[:, 0] + part[:, 1]

    n_candidates = min(len(X), max(n_instances * pool_factor, n_instances + 1))
    candidate_idx = np.argsort(margin)[:n_candidates]

    if n_instances == 1:
        return candidate_idx[:1]

    X_cand = np.asarray(X, dtype=float)[candidate_idx]
    n_unique = len(np.unique(X_cand, axis=0))
    if n_unique <= n_instances:
        return candidate_idx[:n_instances]

    kmeans = KMeans(n_clusters=n_instances, n_init=10, random_state=get_seed())
    kmeans.fit(X_cand)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_cand)

    selected = list(dict.fromkeys(candidate_idx[closest]))
    for idx in candidate_idx:
        if len(selected) >= n_instances:
            break
        if idx not in selected:
            selected.append(int(idx))

    return np.array(selected[:n_instances], dtype=int)