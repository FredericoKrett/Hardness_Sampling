import os
import csv
import math
import traceback
import warnings
from pathlib import Path
from multiprocessing import Pool
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import config
from active_learning_experiment import ActiveLearningExperiment


RANDOM_STATE = 42

_DRIVE_ROOT = Path("/content/drive/MyDrive/hardness_sampling_hier_mcm")
REPO_ROOT = _DRIVE_ROOT if _DRIVE_ROOT.exists() else Path(__file__).resolve().parent.parent

RESULTS_ROOT = REPO_ROOT / "results"
FAILURE_LOG_DIR = REPO_ROOT / "logs"


def _init_worker():
    warnings.filterwarnings("ignore")


def get_results_dir(batch_size, initial_size):
    if initial_size == 5:
        results_dir = RESULTS_ROOT / f"svc_batch{batch_size}"
    else:
        results_dir = RESULTS_ROOT / f"init{initial_size}" / f"svc_batch{batch_size}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_result_file(dataset_file, classifier_name, query_strategy, batch_size, initial_size):
    results_dir = get_results_dir(batch_size, initial_size)
    dataset_name = Path(dataset_file).stem

    filename = "_".join([
        dataset_name,
        f"{config.N_RUNS}x{config.N_SPLITS}",
        classifier_name,
        query_strategy.__name__
    ]) + ".csv"

    return results_dir / filename


def inspect_dataset(dataset_file, batch_size, initial_size):
    data = pd.read_csv(dataset_file)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    n_classes = len(np.unique(y))
    effective_init = max(initial_size, n_classes)
    total_queries = config.QUERIES_BY_INIT[initial_size]
    n_queries = total_queries // batch_size

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    total_rows = 0
    for train_index, _ in skf.split(X, y):
        train_size = len(train_index)
        # Ignora datasets menores que L0 + total_queries no fold de treino
        if train_size < effective_init + total_queries:
            return False, 0

        pool_size = train_size - effective_init
        executed_queries = min(n_queries, math.ceil(pool_size / batch_size))
        total_rows += 1 + executed_queries

    return True, total_rows * config.N_RUNS


def csv_is_complete(result_file, expected_rows):
    if not result_file.exists():
        return False
    if result_file.stat().st_size == 0:
        return False

    required_columns = {"time", "dataset", "classifier", "method",
                        "run", "fold", "query", "kappa"}

    try:
        with open(result_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                return False
            if not required_columns.issubset(set(reader.fieldnames)):
                return False
            rows = list(reader)

        if len(rows) != expected_rows:
            return False

        df = pd.DataFrame(rows)
        df["run"] = df["run"].astype(int)
        df["fold"] = df["fold"].astype(int)
        df["query"] = df["query"].astype(int)
        df["kappa"] = df["kappa"].astype(float)

        if df.duplicated(subset=["run", "fold", "query"]).any():
            return False

        for (_, _), group in df.groupby(["run", "fold"]):
            queries = sorted(group["query"].tolist())
            if not queries or queries[0] != 0:
                return False
            expected_queries = list(range(queries[-1] + 1))
            if queries != expected_queries:
                return False

        return True

    except Exception:
        return False


def resolve_methods(initial_size, method_names=None):
    base_methods = (
        config.ALL_SAMPLING_METHODS
        if initial_size == 5
        else config.TOP_SAMPLING_METHODS
    )
    if not method_names:
        return base_methods

    all_known = {
        fn.__name__: fn
        for fn in list(config.ALL_SAMPLING_METHODS) + list(config.TOP_SAMPLING_METHODS)
    }
    selected = []
    for name in method_names:
        if name not in all_known:
            raise ValueError(f"Método '{name}' não encontrado em config.py.")
        selected.append(all_known[name])
    return selected


def prepare_experiments(batch_size, initial_size, method_names=None):
    csv_dir = REPO_ROOT / "datasets" / "csv"
    datasets = sorted(str(file) for file in csv_dir.glob("*.csv"))
    if not datasets:
        raise RuntimeError(f"Nenhum dataset CSV encontrado em {csv_dir}.")

    classifier_name = "SVC"
    repair_jobs = []
    new_jobs = []
    valid_datasets = 0
    methods = resolve_methods(initial_size, method_names)

    for dataset_file in datasets:
        is_valid, expected_rows = inspect_dataset(dataset_file, batch_size, initial_size)
        if not is_valid:
            continue

        valid_datasets += 1

        for query_strategy in methods:
            result_file = get_result_file(dataset_file, classifier_name,
                                          query_strategy, batch_size, initial_size)
            if csv_is_complete(result_file, expected_rows):
                continue

            job = (dataset_file, classifier_name, query_strategy, batch_size, initial_size)
            if result_file.exists():
                repair_jobs.append(job)
            else:
                new_jobs.append(job)

    return repair_jobs, new_jobs, valid_datasets, len(datasets), methods


def run_experiment(args):
    dataset_file, classifier_name, query_strategy, batch_size, initial_size = args
    estimator = config.CLASSIFIER_DICT[classifier_name]
    results_dir = get_results_dir(batch_size, initial_size)
    total_queries = config.QUERIES_BY_INIT[initial_size]
    n_queries = total_queries // batch_size

    try:
        experiment = ActiveLearningExperiment(
            dataset_file,
            estimator=estimator,
            query_strategy=query_strategy,
            n_queries=n_queries,
            n_runs=config.N_RUNS,
            n_folds=config.N_SPLITS,
            initial_labeled_size=initial_size,
            batch_size=batch_size,
            results_dir=str(results_dir),
            random_state=RANDOM_STATE,
            estimator_name=classifier_name,
        )
        experiment.run_strategy()
        return (True, dataset_file, query_strategy.__name__, batch_size, "")
    except Exception:
        return (False, dataset_file, query_strategy.__name__, batch_size, traceback.format_exc())


def run_jobs_and_report(jobs, desc):
    failures = []
    with Pool(config.N_WORKERS, initializer=_init_worker) as pool:
        for ok, dataset_file, method_name, batch, error in tqdm(
            pool.imap_unordered(run_experiment, jobs),
            total=len(jobs),
            desc=desc,
        ):
            if not ok:
                failures.append((dataset_file, method_name, batch, error))
                print(f"[ERRO] {Path(dataset_file).stem} | {method_name} | batch{batch}")
    return failures


def save_failure_log(failures, batch_size, initial_size):
    if not failures:
        return
    FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = FAILURE_LOG_DIR / f"failures_init{initial_size}_batch{batch_size}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        for dataset_file, method_name, batch, error in failures:
            f.write(f"{'='*70}\n{Path(dataset_file).stem} | {method_name} | batch{batch}\n")
            f.write(error + "\n")
    print(f"\n{len(failures)} falha(s) registradas em: {log_path}")


def execute_single_setup(batch_size, initial_size, method_names=None):
    total_queries = config.QUERIES_BY_INIT[initial_size]
    results_dir = get_results_dir(batch_size, initial_size)
    existing_files = list(results_dir.glob("*.csv"))

    repair_jobs, new_jobs, valid_ds, total_ds, methods = prepare_experiments(
        batch_size, initial_size, method_names
    )

    print()
    print("=" * 70)
    print(f"EXPERIMENTO SVC | L0 = {initial_size} | BATCH = {batch_size} | N_QUERIES = {total_queries}")
    print("=" * 70)
    print(f"Repo root:        {REPO_ROOT}  (existe: {REPO_ROOT.exists()})")
    print(f"Batch:            {batch_size}")
    print(f"Initial size L0:  {initial_size}")
    print(f"Métodos:          {len(methods)}")
    print(f"Datasets válidos: {valid_ds}/{total_ds} (mín. {initial_size + total_queries} no treino)")
    print(f"Workers:          {config.N_WORKERS}  (núcleos: {os.cpu_count()})")
    print(f"Resultados:       {results_dir}")
    print(f"CSVs na pasta:    {len(existing_files)}")
    print(f"CSV incompletos:  {len(repair_jobs)}")
    print(f"CSV novos:        {len(new_jobs)}")
    print("=" * 70)
    print()

    all_failures = []

    if repair_jobs:
        print("Refazendo CSVs incompletos...")
        for job in repair_jobs:
            dataset_file, classifier_name, query_strategy, batch, init_sz = job
            result_file = get_result_file(dataset_file, classifier_name, query_strategy, batch, init_sz)
            if result_file.exists():
                result_file.unlink()

        all_failures += run_jobs_and_report(
            repair_jobs, f"Refazendo L0={initial_size} batch={batch_size}"
        )

    if new_jobs:
        print("Executando novos experimentos...")
        all_failures += run_jobs_and_report(
            new_jobs, f"Executando L0={initial_size} batch={batch_size}"
        )

    save_failure_log(all_failures, batch_size, initial_size)

    print("=" * 70)
    if not repair_jobs and not new_jobs:
        print(f"L0={initial_size} | BATCH {batch_size} JÁ ESTÁ COMPLETO")
    elif all_failures:
        print(f"L0={initial_size} | BATCH {batch_size} FINALIZADO COM {len(all_failures)} FALHA(S)")
    else:
        print(f"L0={initial_size} | BATCH {batch_size} FINALIZADO SEM FALHAS")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", "--batches",
        dest="batches",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25],
        choices=[1, 5, 10, 25],
    )
    parser.add_argument(
        "--initial-size", "--initial-sizes",
        dest="initial_sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        choices=[5, 100, 500, 1000],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.N_WORKERS,
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers deve ser pelo menos 1.")
    config.N_WORKERS = args.workers

    for init_sz in args.initial_sizes:
        for b_sz in args.batches:
            execute_single_setup(
                batch_size=b_sz,
                initial_size=init_sz,
                method_names=args.methods,
            )