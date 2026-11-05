import numpy as np
from sklearn.neighbors import NearestNeighbors


def kdn_adj(classifier, X_pool, n_instances=1, k=5):
    """
    Label-Agnostic k-Disagreeing Neighbors (kDN_adj) baseado em entropia
    (Coutinho, Ueda & Lorena - Eq. 5).
    """
    X_pool_arr = np.asarray(X_pool)
    n_pool = len(X_pool_arr)
    if n_instances >= n_pool:
        return np.arange(n_pool)

    # Conjunto rotulado D_l atual do ActiveLearner (modAL)
    X_labeled = np.asarray(classifier.X_training)
    y_labeled = np.asarray(classifier.y_training)

    # k_adj = min(k, |D_l|) conforme Seção 3.1 do artigo
    k_adj = min(k, len(X_labeled))

    # Busca os k_adj vizinhos mais próximos de cada instância de X_pool dentro de X_labeled
    nn = NearestNeighbors(n_neighbors=k_adj, metric="euclidean", n_jobs=1)
    nn.fit(X_labeled)
    _, neighbor_indices = nn.kneighbors(X_pool_arr)

    # Rótulos dos k_adj vizinhos: matriz de formato (n_pool, k_adj)
    neighbor_labels = y_labeled[neighbor_indices]

    # Entropia de Shannon das proporções de classe na vizinhança
    classes = np.unique(y_labeled)
    entropies = np.zeros(n_pool, dtype=float)

    for c in classes:
        p_c = np.mean(neighbor_labels == c, axis=1)
        mask = p_c > 0
        entropies[mask] -= p_c[mask] * np.log(p_c[mask])

    # Seleciona as n_instances com maior entropia (maior dificuldade)
    # Desempate determinístico mantendo a ordem dos maiores valores
    query_idx = np.argsort(entropies, kind="stable")[-n_instances:][::-1]
    return query_idx