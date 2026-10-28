import os
import csv
import math
import traceback
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import config
from active_learning_experiment import ActiveLearningExperiment


# ============================================================
# CONFIGURAÇÃO AUXILIAR
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"

RANDOM_STATE = 42

REPO_ROOT = Path("/content/drive/MyDrive/hardness_sampling_hier_mcm")

RESULTS_ROOT = REPO_ROOT / "results"
FAILURE_LOG_DIR = REPO_ROOT / "logs"


# ============================================================
# PASTA DOS RESULTADOS
# ============================================================

def get_results_dir(batch_size, initial_size):
    if initial_size == 5:
        results_dir = RESULTS_ROOT / f"svc_batch{batch_size}"
    else:
        results_dir = RESULTS_ROOT / f"init{initial_size}" / f"svc_batch{batch_size}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================================
# NOME DO ARQUIVO DE RESULTADO
# ============================================================

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


# ============================================================
# FILTRO DE DATASETS E NÚMERO ESPERADO DE LINHAS
# ============================================================

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
    for train_index, test_index in skf.split(X, y):
        train_size = len(train_index)
        # Filtra datasets onde o fold de treino nao comporta L0 + total_queries
        if train_size < effective_init + total_queries:
            return False, 0

        pool_size = train_size - effective_init
        executed_queries = min(n_queries, math.ceil(pool_size / batch_size))
        total_rows += 1 + executed_queries  # query 0 + cada query realizada

    return True, total_rows * config.N_RUNS


# ============================================================
# VERIFICAÇÃO DO CSV
# ============================================================

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

        for (run, fold), group in df.groupby(["run", "fold"]):
            queries = sorted(group["query"].tolist())
            if not queries:
                return False
            if queries[0] != 0:
                return False
            expected_queries = list(range(queries[-1] + 1))
            if queries != expected_queries:
                return False

        return True

    except Exception:
        return False


# ============================================================
# PREPARAÇÃO DOS EXPERIMENTOS
# ============================================================

def prepare_experiments(batch_size, initial_size):
    csv_dir = REPO_ROOT / "datasets" / "csv"
    datasets = sorted(str(file) for file in csv_dir.glob("*.csv"))
    if not datasets:
        raise RuntimeError(f"Nenhum dataset CSV foi encontrado em {csv_dir}.")

    classifier_name = "SVC"
    repair_jobs = []
    new_jobs = []
    valid_datasets = 0

    for dataset_file in datasets:
        is_valid, expected_rows = inspect_dataset(dataset_file, batch_size, initial_size)
        if not is_valid:
            continue

        valid_datasets += 1

        for query_strategy in config.SAMPLING_METHODS:
            result_file = get_result_file(dataset_file, classifier_name,
                                          query_strategy, batch_size, initial_size)
            complete = csv_is_complete(result_file, expected_rows)

            if complete:
                continue

            job = (dataset_file, classifier_name, query_strategy, batch_size, initial_size)
            if result_file.exists():
                repair_jobs.append(job)
            else:
                new_jobs.append(job)

    return repair_jobs, new_jobs, valid_datasets, len(datasets)


# ============================================================
# EXECUTA UM EXPERIMENTO
# ============================================================

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


def run_jobs_and_report(jobs, batch_size, desc):
    failures = []
    with Pool(config.N_WORKERS) as pool:
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
    print("Rode o script novamente para tentar refazer essas combinações.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Executa um batch do experimento SVC com filtragem por tamanho inicial."
    )
    parser.add_argument("--batch-size", type=int, required=True, choices=[1, 5, 10, 25],
                        help="Batch size: 1, 5, 10 ou 25.")
    parser.add_argument("--initial-size", type=int, default=100, choices=[5, 100, 500, 1000],
                        help="Tamanho inicial rotulado L0: 5, 100, 500 ou 1000.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Processos paralelos.")
    args = parser.parse_args()
    batch_size = args.batch_size
    initial_size = args.initial_size
    total_queries = config.QUERIES_BY_INIT[initial_size]

    if args.workers < 1:
        raise ValueError("--workers deve ser pelo menos 1.")
    config.N_WORKERS = args.workers

    results_dir = get_results_dir(batch_size, initial_size)
    existing_files = list(results_dir.glob("*.csv"))

    repair_jobs, new_jobs, valid_ds, total_ds = prepare_experiments(batch_size, initial_size)

    print()
    print("=" * 70)
    print(f"EXPERIMENTO SVC | L0 = {initial_size} | N_QUERIES = {total_queries}")
    print("=" * 70)
    print(f"Repo root:        {REPO_ROOT}  (existe: {REPO_ROOT.exists()})")
    print(f"Batch:            {batch_size}")
    print(f"Initial size L0:  {initial_size}")
    print(f"Datasets válidos: {valid_ds}/{total_ds} (mínimo de {initial_size + total_queries} instâncias no treino)")
    print(f"Workers:          {config.N_WORKERS}  (núcleos detectados: {os.cpu_count()})")
    print(f"Resultados:       {results_dir}")
    print(f"CSVs já na pasta: {len(existing_files)}")
    print(f"CSV incompletos:  {len(repair_jobs)}")
    print(f"CSV novos:        {len(new_jobs)}")
    print("=" * 70)
    print()

    all_failures = []

    if repair_jobs:
        print("=" * 70)
        print("REFAZENDO CSVs INCOMPLETOS")
        print("=" * 70)

        for job in repair_jobs:
            dataset_file, classifier_name, query_strategy, batch, init_sz = job
            result_file = get_result_file(dataset_file, classifier_name, query_strategy, batch, init_sz)
            print(f"Excluindo: {result_file.name}")
            result_file.unlink()
        print()

        all_failures += run_jobs_and_report(
            repair_jobs, batch_size, f"Refazendo L0={initial_size} batch={batch_size}"
        )
        print("\nCSV incompletos processados.\n")

    if new_jobs:
        print("=" * 70)
        print("EXECUTANDO NOVOS EXPERIMENTOS")
        print("=" * 70)

        all_failures += run_jobs_and_report(
            new_jobs, batch_size, f"Executando L0={initial_size} batch={batch_size}"
        )
        print("\nNovos experimentos processados.\n")

    save_failure_log(all_failures, batch_size, initial_size)

    print("=" * 70)
    if not repair_jobs and not new_jobs:
        print(f"L0={initial_size} | BATCH {batch_size} JÁ ESTÁ COMPLETO")
    elif all_failures:
        print(f"L0={initial_size} | BATCH {batch_size} FINALIZADO COM {len(all_failures)} FALHA(S)")
    else:
        print(f"L0={initial_size} | BATCH {batch_size} FINALIZADO SEM FALHAS")
    print(f"Resultados: {results_dir}")
    print("=" * 70)