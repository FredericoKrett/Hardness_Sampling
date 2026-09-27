from functools import partial

from modAL.uncertainty import margin_sampling
from sklearn.svm import SVC

import strategies.hardness as ih
from strategies.random import random_sampling
from strategies.expected_error import expected_error_reduction
from strategies.information_density import (density_weighted_sampling,
                                            training_utility_sampling)


# total number of instances queried during the active learning process
N_QUERIES = 100

# number of instances selected in each active learning iteration
BATCH_SIZES = [1, 5, 10, 25]

# n_splits for cross-validation
N_SPLITS = 5

# number of times cross-validation will be executed
N_RUNS = 1

RESULTS_DIR = '../results/svc_batches'

CLASSIFIER_DICT = {
    "SVC": partial(SVC, probability=True),
}

SAMPLING_METHODS = [
    random_sampling,
    margin_sampling,
    density_weighted_sampling,
    training_utility_sampling,
    expected_error_reduction,
    ih.borderline_points_sampling,
    ih.class_balance_sampling,
    ih.class_likelihood_sampling,
    ih.class_likeliood_diff_sampling,
    ih.disjunct_class_percentage_sampling,
    ih.disjunct_size_sampling,
    ih.f1_sampling,
    ih.f2_sampling,
    ih.f3_sampling,
    ih.f4_sampling,
    ih.harmfulness_sampling,
    ih.intra_extra_ratio_sampling,
    ih.k_disagreeing_neighbors_sampling,
    ih.local_set_cardinality_sampling,
    ih.ls_radius_sampling,
    ih.minority_value_sampling,
    ih.tree_depth_pruned_sampling,
    ih.tree_depth_unpruned_sampling,
    ih.usefulness_sampling,
]

ARFF_DIR = '../datasets/arff/'
CSV_DIR = '../datasets/csv'

N_WORKERS = 48

LOG_DIR = 'logs/'