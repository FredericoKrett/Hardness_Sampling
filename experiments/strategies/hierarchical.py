import numpy as np
from modAL import ActiveLearner
from pyhard import get_seed
from sklearn.cluster import AgglomerativeClustering


class _HierTreeSampler:
    """Estrutura interna da árvore de Dasgupta & Hsu (2008)."""

    def __init__(self, X_full: np.ndarray, beta: float = 2.0, random_state: int = 42):
        self.n = len(X_full)
        self.beta = beta
        self.rng = np.random.default_rng(random_state)

        clustering = AgglomerativeClustering(
            n_clusters=2, linkage="ward", compute_full_tree=True
        ).fit(np.asarray(X_full, dtype=float))

        self.children = clustering.children_
        self.n_nodes = 2 * self.n - 1
        self.root = self.n_nodes - 1

        self.weights = np.ones(self.n_nodes, dtype=float)
        self.sub_leaves = [[i] for i in range(self.n)]
        for i, (left, right) in enumerate(self.children):
            v = self.n + i
            self.weights[v] = self.weights[left] + self.weights[right]
            self.sub_leaves.append(self.sub_leaves[left] + self.sub_leaves[right])
        self.weights /= self.n

        self.classes_ = np.array([])
        self.admissible = np.zeros((self.n_nodes, 0), dtype=bool)
        self.pruning = [self.root]

    def _sync_classes(self, y_labeled: np.ndarray) -> np.ndarray:
        unique_y = np.unique(y_labeled)
        new_classes = np.setdiff1d(unique_y, self.classes_)
        if len(new_classes) > 0:
            self.classes_ = np.concatenate([self.classes_, new_classes])
            pad = np.zeros((self.n_nodes, len(new_classes)), dtype=bool)
            self.admissible = np.hstack([self.admissible, pad])
        return np.searchsorted(self.classes_, y_labeled)

    def _update_tree(self, labeled_leaves: np.ndarray, y_labeled: np.ndarray):
        y_enc = self._sync_classes(y_labeled)
        n_c = len(self.classes_)

        counts = np.zeros((self.n_nodes, max(2, n_c)), dtype=float)
        counts[labeled_leaves, y_enc] = 1.0
        for i, (left, right) in enumerate(self.children):
            counts[self.n + i] = counts[left] + counts[right]

        n_v = counts.sum(axis=1, keepdims=True)
        p_vl = np.divide(counts, n_v, out=np.zeros_like(counts), where=n_v > 0)

        delta = np.divide(1.0, n_v, out=np.ones_like(n_v), where=n_v > 0) + np.sqrt(
            np.divide(p_vl * (1.0 - p_vl), n_v, out=np.zeros_like(p_vl), where=n_v > 0)
        )
        p_lb = np.maximum(0.0, p_vl - delta)
        p_ub = np.minimum(1.0, p_vl + delta)

        if self.admissible.shape[1] < counts.shape[1]:
            pad = np.zeros((self.n_nodes, counts.shape[1] - self.admissible.shape[1]), dtype=bool)
            self.admissible = np.hstack([self.admissible, pad])

        err_lb = 1.0 - p_lb
        err_ub = 1.0 - p_ub
        for c in range(counts.shape[1]):
            other_err_ub = np.delete(err_ub, c, axis=1).min(axis=1)
            adm_c = (n_v[:, 0] > 0) & (err_lb[:, c] < self.beta * other_err_ub)
            self.admissible[:, c] |= adm_c

        eps_tilde = np.where(self.admissible, 1.0 - p_vl, 1.0)
        best_err = eps_tilde.min(axis=1)
        best_label = np.argmax(p_vl, axis=1)

        scores = best_err.copy()
        split = np.zeros(self.n_nodes, dtype=bool)

        for i, (left, right) in enumerate(self.children):
            v = self.n + i
            if self.admissible[v].any():
                split_err = (
                    self.weights[left] * scores[left] + self.weights[right] * scores[right]
                ) / self.weights[v]
                if split_err < best_err[v]:
                    scores[v] = split_err
                    split[v] = True

        self.pruning = []
        stack = [self.root]
        while stack:
            u = stack.pop()
            if split[u]:
                left, right = self.children[u - self.n]
                stack.extend([right, left])
            else:
                self.pruning.append(u)

        return p_lb, best_label

    def select_leaves(self, labeled_leaves: np.ndarray, y_labeled: np.ndarray,
                      available_leaves: np.ndarray, batch_size: int) -> list:
        p_lb, best_label = self._update_tree(labeled_leaves, y_labeled)
        avail_set = set(map(int, available_leaves))
        chosen_leaves = []

        for _ in range(batch_size):
            valid_clusters = []
            cluster_leaves = []
            for v in self.pruning:
                unlabeled_in_v = [leaf for leaf in self.sub_leaves[v] if leaf in avail_set]
                if unlabeled_in_v:
                    valid_clusters.append(v)
                    cluster_leaves.append(unlabeled_in_v)

            if not valid_clusters:
                break

            probs = np.array([
                self.weights[v] * (1.0 - p_lb[v, best_label[v]]) for v in valid_clusters
            ], dtype=float)

            if probs.sum() <= 0:
                probs = np.array([self.weights[v] for v in valid_clusters], dtype=float)
            probs /= probs.sum()

            c_idx = self.rng.choice(len(valid_clusters), p=probs)
            leaf = int(self.rng.choice(cluster_leaves[c_idx]))

            chosen_leaves.append(leaf)
            avail_set.remove(leaf)

        return chosen_leaves


def hierarchical_sampling(learner: ActiveLearner, X: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """Estratégia Hier (Dasgupta & Hsu, 2008) compatível com modAL."""
    if len(X) <= n_instances:
        return np.arange(len(X), dtype=int)

    y_tr = np.asarray(learner.y_training).ravel()
    n_l = len(y_tr)
    n_u = len(X)

    if not hasattr(learner, "_hier_sampler") or learner._hier_sampler.n != (n_l + n_u):
        X_full = np.vstack([learner.X_training, X])
        learner._hier_sampler = _HierTreeSampler(X_full, random_state=get_seed())
        learner._hier_labeled_leaves = list(range(n_l))
        learner._hier_pool_leaves = np.arange(n_l, n_l + n_u, dtype=int)
        learner._hier_pending_leaves = []
        learner._hier_pending_local = np.array([], dtype=int)

    if len(learner._hier_labeled_leaves) < n_l and len(learner._hier_pending_leaves) > 0:
        learner._hier_labeled_leaves.extend(learner._hier_pending_leaves)
        learner._hier_pool_leaves = np.delete(learner._hier_pool_leaves, learner._hier_pending_local)
        learner._hier_pending_leaves = []
        learner._hier_pending_local = np.array([], dtype=int)

    if len(learner._hier_pending_local) == n_instances and len(learner._hier_labeled_leaves) == n_l:
        return learner._hier_pending_local

    chosen_leaves = learner._hier_sampler.select_leaves(
        labeled_leaves=np.array(learner._hier_labeled_leaves[:n_l], dtype=int),
        y_labeled=y_tr,
        available_leaves=learner._hier_pool_leaves,
        batch_size=n_instances,
    )

    leaf_to_local = {int(leaf): idx for idx, leaf in enumerate(learner._hier_pool_leaves)}
    local_indices = np.array([leaf_to_local[leaf] for leaf in chosen_leaves], dtype=int)

    learner._hier_pending_leaves = chosen_leaves
    learner._hier_pending_local = local_indices

    return local_indices