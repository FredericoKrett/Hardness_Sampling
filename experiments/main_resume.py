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

REPO_ROOT = Path("/content/drive/MyDrive/hardness_sampling_svc")

RESULTS_ROOT = REPO_ROOT / "results"
FAILURE_LOG_DIR = REPO_ROOT / "logs"


# ============================================================
# PASTA DOS RESULTADOS
# ============================================================

def get_results_dir(batch_size):
    results_dir = RESULTS_ROOT / f"svc_batch{batch_size}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================================
# NOME DO ARQUIVO DE RESULTADO
# ============================================================

def get_result_file(dataset_file, classifier_name, query_strategy, batch_size):
    results_dir = get_results_dir(batch_size)
    dataset_name = Path(dataset_file).stem

    filename = "_".join([
        dataset_name,
        f"{config.N_RUNS}x{config.N_SPLITS}",
        classifier_name,
        query_strategy.__name__
    ]) + ".csv"

    return results_dir / filename


def get_initial_labeled_size():
    return 5  # mesmo default do ActiveLearningExperiment original


# ============================================================
# NÚMERO ESPERADO DE LINHAS
# ============================================================

def get_expected_rows(dataset_file, batch_size):
    data = pd.read_csv(dataset_file)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    initial_labeled_size = get_initial_labeled_size()
    n_queries = config.N_QUERIES // batch_size

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    total_rows = 0
    for train_index, test_index in skf.split(X, y):
        train_size = len(train_index)
        pool_size = train_size - initial_labeled_size

        if pool_size <= 0:
            executed_queries = 0
        else:
            executed_queries = min(n_queries, math.ceil(pool_size / batch_size))

        total_rows += 1 + executed_queries  # query 0 + cada query realizada

    return total_rows * config.N_RUNS


# ============================================================
# VERIFICAÇÃO DO CSV
# ============================================================

def csv_is_complete(result_file, dataset_file, batch_size):
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

        expected = get_expected_rows(dataset_file, batch_size)
        if len(rows) != expected:
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

def prepare_experiments(batch_size):
    csv_dir = REPO_ROOT / "datasets" / "csv"
    datasets = sorted(str(file) for file in csv_dir.glob("*.csv"))
    if not datasets:
        raise RuntimeError(f"Nenhum dataset CSV foi encontrado em {csv_dir}.")

    classifier_name = "SVC"
    repair_jobs = []
    new_jobs = []

    for dataset_file in datasets:
        for query_strategy in config.SAMPLING_METHODS:
            result_file = get_result_file(dataset_file, classifier_name,
                                          query_strategy, batch_size)
            complete = csv_is_complete(result_file, dataset_file, batch_size)

            if complete:
                continue

            if result_file.exists():
                repair_jobs.append((dataset_file, classifier_name, query_strategy, batch_size))
            else:
                new_jobs.append((dataset_file, classifier_name, query_strategy, batch_size))

    return repair_jobs, new_jobs


# ============================================================
# EXECUTA UM EXPERIMENTO
# ============================================================

def run_experiment(args):
    dataset_file, classifier_name, query_strategy, batch_size = args
    estimator = config.CLASSIFIER_DICT[classifier_name]
    results_dir = get_results_dir(batch_size)
    n_queries = config.N_QUERIES // batch_size

    try:
        experiment = ActiveLearningExperiment(
            dataset_file,
            estimator=estimator,
            query_strategy=query_strategy,
            n_queries=n_queries,
            n_runs=config.N_RUNS,
            n_folds=config.N_SPLITS,
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
    """Roda um Pool sobre `jobs` sem deixar uma falha derrubar as demais,
    e devolve a lista de falhas para reportar/registrar no final."""
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


def save_failure_log(failures, batch_size):
    if not failures:
        return
    FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = FAILURE_LOG_DIR / f"failures_batch{batch_size}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        for dataset_file, method_name, batch, error in failures:
            f.write(f"{'='*70}\n{Path(dataset_file).stem} | {method_name} | batch{batch}\n")
            f.write(error + "\n")
    print(f"\n{len(failures)} falha(s) registradas em: {log_path}")
    print("Rode o script novamente para tentar refazer essas combinações "
         "(elas ficaram sem CSV completo, então serão reconhecidas como pendentes).")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Executa um batch do experimento SVC com recuperação de resultados."
    )
    parser.add_argument("--batch-size", type=int, required=True, choices=[1, 5, 10, 25],
                        help="Batch size: 1, 5, 10 ou 25.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Processos paralelos (default: núcleos reais da máquina, "
                             "via os.cpu_count(); ignora config.N_WORKERS).")
    args = parser.parse_args()
    batch_size = args.batch_size

    if args.workers < 1:
        raise ValueError("--workers deve ser pelo menos 1.")
    config.N_WORKERS = args.workers

    results_dir = get_results_dir(batch_size)
    existing_files = list(results_dir.glob("*.csv"))

    print()
    print("=" * 70)
    print("EXPERIMENTO SVC")
    print("=" * 70)
    print(f"Repo root:  {REPO_ROOT}  (existe: {REPO_ROOT.exists()})")
    print(f"Batch:      {batch_size}")
    print(f"Workers:    {config.N_WORKERS}  (núcleos detectados: {os.cpu_count()})")
    print(f"Resultados: {results_dir}")
    print(f"CSVs já presentes nessa pasta (antes de checar completude): {len(existing_files)}")
    print("=" * 70)
    print()

    if len(existing_files) == 0:
        print("[ATENÇÃO] A pasta de resultados parece vazia. Se você esperava "
             "encontrar arquivos de uma execução anterior, PARE agora e confira "
             "se o Google Drive terminou de montar/sincronizar antes de continuar "
             "-- senão o experimento vai reprocessar tudo do zero.\n")

    repair_jobs, new_jobs = prepare_experiments(batch_size)

    print(f"CSV incompletos: {len(repair_jobs)}")
    print(f"CSV novos:       {len(new_jobs)}")
    print()

    all_failures = []

    if repair_jobs:
        print("=" * 70)
        print("REFAZENDO CSVs INCOMPLETOS")
        print("=" * 70)

        for job in repair_jobs:
            dataset_file, classifier_name, query_strategy, batch = job
            result_file = get_result_file(dataset_file, classifier_name, query_strategy, batch)
            print(f"Excluindo: {result_file.name}")
            result_file.unlink()
        print()

        all_failures += run_jobs_and_report(repair_jobs, batch_size, f"Refazendo batch {batch_size}")
        print("\nCSV incompletos processados.\n")

    if new_jobs:
        print("=" * 70)
        print("EXECUTANDO NOVOS EXPERIMENTOS")
        print("=" * 70)

        all_failures += run_jobs_and_report(new_jobs, batch_size, f"Executando batch {batch_size}")
        print("\nNovos experimentos processados.\n")

    save_failure_log(all_failures, batch_size)

    print("=" * 70)
    if not repair_jobs and not new_jobs:
        print(f"BATCH {batch_size} JÁ ESTÁ COMPLETO")
    elif all_failures:
        print(f"BATCH {batch_size} FINALIZADO COM {len(all_failures)} FALHA(S) -- veja o log acima")
    else:
        print(f"BATCH {batch_size} FINALIZADO SEM FALHAS")
    print(f"Resultados: {results_dir}")
    print("=" * 70)