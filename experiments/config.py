from functools import partial

from modAL.uncertainty import margin_sampling
from sklearn.svm import SVC

from strategies.expected_error import expected_error_reduction
from strategies.hardness import (
    borderline_points_sampling,
    class_balance_sampling,
    class_likelihood_sampling,
    class_likeliood_diff_sampling,
    disjunct_class_percentage_sampling,
    disjunct_size_sampling,
    f1_sampling,
    f2_sampling,
    f3_sampling,
    f4_sampling,
    harmfulness_sampling,
    intra_extra_ratio_sampling,
    k_disagreeing_neighbors_sampling,
    local_set_cardinality_sampling,
    ls_radius_sampling,
    minority_value_sampling,
    tree_depth_pruned_sampling,
    tree_depth_unpruned_sampling,
    usefulness_sampling,
)
from strategies.hierarchical import hierarchical_sampling
from strategies.hier_kdn import hier_kdn_sampling
from strategies.information_density import (
    density_weighted_sampling,
    training_utility_sampling,
)
from strategies.kdn_adj import kdn_adj
from strategies.mcm import mcm_sampling
from strategies.random import random_sampling

# total number of instances queried during the active learning process
N_QUERIES = 100
N_QUERIES_REDUCED = 50
MAX_QUERIES = 50

# mapeamento automatico de queries conforme o tamanho inicial L0
QUERIES_BY_INIT = {
    5: N_QUERIES,           # 100 queries (configurcao padrao)
    100: N_QUERIES_REDUCED, # 50 queries
    500: N_QUERIES_REDUCED, # 50 queries
    1000: N_QUERIES_REDUCED # 50 queries
}

# number of instances selected in each active learning iteration
BATCH_SIZES = [1, 5, 10, 25]
BATCH_SIZE = 1

# n_splits for cross-validation
N_SPLITS = 5
N_FOLDS = 5

# number of times cross-validation will be executed
N_RUNS = 1

INITIAL_SIZE = 5

RESULTS_DIR = '../results/svc_batches'

CLASSIFIER_DICT = {
    "SVC": partial(SVC, probability=True),
}

ARFF_DIR = '../datasets/arff/'
CSV_DIR = '../datasets/csv'
LOG_DIR = 'logs/'

N_WORKERS = 48

# Lista completa (28 estratégias) para L0 = 5
ALL_SAMPLING_METHODS = [
    random_sampling,
    margin_sampling,
    expected_error_reduction,
    density_weighted_sampling,
    training_utility_sampling,
    k_disagreeing_neighbors_sampling,
    kdn_adj,
    hier_kdn_sampling,
    disjunct_size_sampling,
    disjunct_class_percentage_sampling,
    tree_depth_pruned_sampling,
    tree_depth_unpruned_sampling,
    class_likelihood_sampling,
    class_likeliood_diff_sampling,
    f1_sampling,
    f2_sampling,
    f3_sampling,
    f4_sampling,
    intra_extra_ratio_sampling,
    borderline_points_sampling,
    local_set_cardinality_sampling,
    ls_radius_sampling,
    usefulness_sampling,
    harmfulness_sampling,
    minority_value_sampling,
    class_balance_sampling,
    hierarchical_sampling,
    mcm_sampling,
]

# Top 10 consolidado via Friedman-Nemenyi + kdn_adj + hier_kdn (para L0 = 100, 500, 1000)
TOP_SAMPLING_METHODS = [
    hierarchical_sampling,
    random_sampling,
    k_disagreeing_neighbors_sampling,
    disjunct_size_sampling,
    mcm_sampling,
    harmfulness_sampling,
    local_set_cardinality_sampling,
    class_likeliood_diff_sampling,
    borderline_points_sampling,
    ls_radius_sampling,
    kdn_adj,
    hier_kdn_sampling,
]

SAMPLING_METHODS = ALL_SAMPLING_METHODS
