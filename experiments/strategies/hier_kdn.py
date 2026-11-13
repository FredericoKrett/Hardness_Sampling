import numpy as np
import pandas as pd
from modAL import ActiveLearner
from pyhard import get_seed

from .hardness import gower_distance
from .hierarchical import _HierTreeSampler


def _compute_kdn_scores_cached(
    learner: ActiveLearner, X_pool: np.ndarray, pool_leaves: np.ndarray, k: int = 10
) -> dict:
    """
    Calcula a medida de dificuldade k-Disagreeing Neighbors (kDN) para cada instância
    disponível no pool de busca, utilizando as predições atuais do classificador.
    Mantém a matriz de distâncias de Gower completa em cache por fold para eficiência.
    """
    n_pool = len(X_pool)
    if n_pool <= 1:
        return {int(leaf): 0.0 for leaf in pool_leaves}

    k_eff = min(k, n_pool - 1)
    y_pred = learner.predict(X_pool)

    # Calcula a matriz de distância de Gower na 1ª query do fold e reutiliza por fatiamento
    if not hasattr(learner, "_gower_full_matrix"):
        X_full = learner._hier_X_full
        learner._gower_full_matrix = gower_distance(pd.DataFrame(X_full))

    dist_sub = learner._gower_full_matrix[np.ix_(pool_leaves, pool_leaves)].copy()
    np.fill_diagonal(dist_sub, -1.0)
    indices_gower = np.argsort(dist_sub, axis=1)[:, 1 : k_eff + 1]

    # Proporção dos k vizinhos mais próximos cuja classe predita diverge da instância avaliada
    neighbor_preds = y_pred[indices_gower]
    kdn_scores = np.mean(neighbor_preds != y_pred[:, None], axis=1)

    return {int(leaf): float(score) for leaf, score in zip(pool_leaves, kdn_scores)}


class _HierKDNTreeSampler(_HierTreeSampler):
    """
    Árvore de agrupamento hierárquico com poda adaptativa (Dasgupta & Hsu, 2008)
    combinada com seleção intra-cluster guiada por dificuldade de instância (kDN).
    """

    def select_leaves_kdn(
        self,
        labeled_leaves: np.ndarray,
        y_labeled: np.ndarray,
        available_leaves: np.ndarray,
        batch_size: int,
        leaf_kdn_scores: dict,
    ) -> list:
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

            # Probabilidade de amostragem do cluster v proporcional a w_v * (1 - p_lb(v, y_hat_v))
            probs = np.array(
                [self.weights[v] * (1.0 - p_lb[v, best_label[v]]) for v in valid_clusters],
                dtype=float,
            )

            if probs.sum() <= 0:
                probs = np.array([self.weights[v] for v in valid_clusters], dtype=float)
            probs /= probs.sum()

            # Passo 1 (Exploração - Hier): Sorteia o cluster ativo na poda atual da árvore
            c_idx = self.rng.choice(len(valid_clusters), p=probs)
            candidates = cluster_leaves[c_idx]

            # Passo 2 (Explotação - kDN): Seleciona no cluster a folha não rotulada de maior kDN
            cand_scores = np.array([leaf_kdn_scores.get(leaf, 0.0) for leaf in candidates])
            max_score = cand_scores.max()
            best_candidates = [
                candidates[i]
                for i in range(len(candidates))
                if np.isclose(cand_scores[i], max_score, atol=1e-9)
            ]
            leaf = int(self.rng.choice(best_candidates))

            chosen_leaves.append(leaf)
            avail_set.remove(leaf)

        return chosen_leaves


def hier_kdn_sampling(
    learner: ActiveLearner, X: np.ndarray, n_instances: int = 1, k: int = 10
) -> np.ndarray:
    """
    Estratégia híbrida de Aprendizado Ativo Hier+kDN compatível com modAL.
    Combina a amostragem hierárquica guiada por poda (Dasgupta & Hsu, 2008) para seleção
    de sub-regiões com a medida k-Disagreeing Neighbors (kDN) para escolha da instância.
    """
    if len(X) <= n_instances:
        return np.arange(len(X), dtype=int)

    y_tr = np.asarray(learner.y_training).ravel()
    n_l = len(y_tr)
    n_u = len(X)

    # Inicializa a árvore hierárquica sobre o conjunto completo (treino inicial + pool) no fold
    if not hasattr(learner, "_hier_kdn_sampler") or learner._hier_kdn_sampler.n != (n_l + n_u):
        X_full = np.vstack([learner.X_training, X])
        learner._hier_X_full = X_full
        learner._hier_kdn_sampler = _HierKDNTreeSampler(X_full, random_state=get_seed())
        learner._hier_kdn_labeled_leaves = list(range(n_l))
        learner._hier_kdn_pool_leaves = np.arange(n_l, n_l + n_u, dtype=int)
        learner._hier_kdn_pending_leaves = []
        learner._hier_kdn_pending_local = np.array([], dtype=int)
        if hasattr(learner, "_gower_full_matrix"):
            delattr(learner, "_gower_full_matrix")

    # Sincroniza as folhas rotuladas após o passo de ensino (teach) da iteração anterior
    if len(learner._hier_kdn_labeled_leaves) < n_l and len(learner._hier_kdn_pending_leaves) > 0:
        learner._hier_kdn_labeled_leaves.extend(learner._hier_kdn_pending_leaves)
        learner._hier_kdn_pool_leaves = np.delete(
            learner._hier_kdn_pool_leaves, learner._hier_kdn_pending_local
        )
        learner._hier_kdn_pending_leaves = []
        learner._hier_kdn_pending_local = np.array([], dtype=int)

    if (
        len(learner._hier_kdn_pending_local) == n_instances
        and len(learner._hier_kdn_labeled_leaves) == n_l
    ):
        return learner._hier_kdn_pending_local

    # Estima o score kDN para todas as folhas disponíveis no pool atual
    leaf_kdn_scores = _compute_kdn_scores_cached(
        learner=learner,
        X_pool=X,
        pool_leaves=learner._hier_kdn_pool_leaves,
        k=k,
    )

    chosen_leaves = learner._hier_kdn_sampler.select_leaves_kdn(
        labeled_leaves=np.array(learner._hier_kdn_labeled_leaves[:n_l], dtype=int),
        y_labeled=y_tr,
        available_leaves=learner._hier_kdn_pool_leaves,
        batch_size=n_instances,
        leaf_kdn_scores=leaf_kdn_scores,
    )

    leaf_to_local = {int(leaf): idx for idx, leaf in enumerate(learner._hier_kdn_pool_leaves)}
    local_indices = np.array([leaf_to_local[leaf] for leaf in chosen_leaves], dtype=int)

    learner._hier_kdn_pending_leaves = chosen_leaves
    learner._hier_kdn_pending_local = local_indices

    return local_indices
