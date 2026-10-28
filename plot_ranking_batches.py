import os
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

warnings.filterwarnings("ignore")

REPO_ROOT = Path(
    "/content/drive/MyDrive/hardness_sampling_hier_mcm"
)

RESULTS_ROOT = REPO_ROOT / "results"
PLOT_ROOT = REPO_ROOT / "plots_ranking"

BATCH_SIZES = [1, 5, 10, 25]

SMOOTHING_WINDOW = {
    1: 1,
    5: 5,
    10: 3,
    25: 1
}

MAX_QUERIES_BY_INIT = {
    5: 100,
    100: 50,
    500: 50,
    1000: 50
}


# ============================================================
# 26 MÉTODOS
# ============================================================

ABREV_DICT = {

    "random_sampling": "Rnd",

    "margin_sampling": "MS",

    "training_utility_sampling": "TU",

    "density_weighted_sampling": "ID",

    "expected_error_reduction": r"EER$_{ent}$",

    "hierarchical_sampling": "Hier",

    "mcm_sampling": "MCM",

    "borderline_points_sampling": "N1I",

    "k_disagreeing_neighbors_sampling": "kDN",

    "local_set_cardinality_sampling": "LSCI",

    "ls_radius_sampling": "LSR",

    "intra_extra_ratio_sampling": "N2I",

    "harmfulness_sampling": "H",

    "usefulness_sampling": "U",

    "class_likelihood_sampling": "CL",

    "class_likeliood_diff_sampling": "CLD",

    "f1_sampling": "F1I",

    "f2_sampling": "F2I",

    "f3_sampling": "F3I",

    "f4_sampling": "F4I",

    "disjunct_class_percentage_sampling": "DCP",

    "disjunct_size_sampling": "DS",

    "tree_depth_pruned_sampling": r"TD$_p$",

    "tree_depth_unpruned_sampling": r"TD$_u$",

    "class_balance_sampling": "CB",

    "minority_value_sampling": "MV",
}


METHODS = list(ABREV_DICT.keys())

print("Métodos esperados:", len(METHODS))


# ============================================================
# GRUPOS
# ============================================================

CLASSIC = {
    "margin_sampling",
    "training_utility_sampling",
    "density_weighted_sampling",
    "expected_error_reduction",
    "hierarchical_sampling",
    "mcm_sampling",
    "random_sampling"
}

NEIGHBORHOOD = {
    "borderline_points_sampling",
    "k_disagreeing_neighbors_sampling",
    "local_set_cardinality_sampling",
    "ls_radius_sampling",
    "intra_extra_ratio_sampling",
    "harmfulness_sampling",
    "usefulness_sampling"
}

LIKELIHOOD = {
    "class_likelihood_sampling",
    "class_likeliood_diff_sampling"
}

FEATURE_BASED = {
    "f1_sampling",
    "f2_sampling",
    "f3_sampling",
    "f4_sampling"
}

TREE_BASED = {
    "disjunct_class_percentage_sampling",
    "disjunct_size_sampling",
    "tree_depth_pruned_sampling",
    "tree_depth_unpruned_sampling"
}

CLASS_BALANCE = {
    "class_balance_sampling",
    "minority_value_sampling"
}


def get_hm_type(method):

    if method == "random_sampling":
        return "baseline"

    if method in CLASSIC:
        return "classic"

    if method in NEIGHBORHOOD:
        return "neighborhood"

    if method in LIKELIHOOD:
        return "likelihood"

    if method in FEATURE_BASED:
        return "feature"

    if method in TREE_BASED:
        return "tree"

    if method in CLASS_BALANCE:
        return "class-balance"

    raise ValueError(
        f"Método não classificado: {method}"
    )


def get_style(method):

    if method == "random_sampling":
        return "--"

    elif method in CLASSIC:
        return "-"

    elif method in NEIGHBORHOOD:
        return "^-"

    elif method in LIKELIHOOD:
        return "s-"

    elif method in FEATURE_BASED:
        return "*-"

    elif method in TREE_BASED:
        return "h-"

    else:
        return "d-"


# ============================================================
# CORES
# ============================================================

color_list = sns.color_palette(
    cc.glasbey_dark,
    n_colors=len(METHODS)
)

COLOR_DICT = dict(
    zip(
        METHODS,
        color_list
    )
)

COLOR_DICT["random_sampling"] = "firebrick"


# ============================================================
# CAMINHOS POR TAMANHO INICIAL
# ============================================================

def get_results_dir_for_plot(batch_size, initial_size):
    if initial_size == 5:
        return RESULTS_ROOT / f"svc_batch{batch_size}"
    return RESULTS_ROOT / f"init{initial_size}" / f"svc_batch{batch_size}"


def get_plot_dir(initial_size):
    if initial_size == 5:
        return PLOT_ROOT
    return PLOT_ROOT / f"init{initial_size}"


# ============================================================
# CARREGAR CSVs
# ============================================================

def load_results(batch_size, initial_size):

    results_dir = get_results_dir_for_plot(batch_size, initial_size)

    print()
    print("=" * 75)
    print(f"L0 = {initial_size} | BATCH {batch_size}")
    print("=" * 75)
    print(f"Pasta: {results_dir}")

    if not results_dir.exists():

        print("Pasta não encontrada.")
        return None

    csv_files = sorted(
        results_dir.glob("*.csv")
    )

    print(
        f"CSVs encontrados: {len(csv_files)}"
    )

    frames = []

    for csv_file in csv_files:

        try:

            df = pd.read_csv(
                csv_file
            )

            required_columns = {
                "dataset",
                "classifier",
                "method",
                "run",
                "fold",
                "query",
                "kappa"
            }

            if not required_columns.issubset(
                df.columns
            ):
                continue

            frames.append(df)

        except Exception as e:

            print(
                f"[IGNORADO] {csv_file.name}: {e}"
            )

    if not frames:

        print(
            "Nenhum CSV com as colunas esperadas."
        )

        return None

    results_df = pd.concat(
        frames,
        ignore_index=True
    )

    # ========================================================
    # NORMALIZA
    # ========================================================

    results_df["classifier"] = (
        results_df["classifier"]
        .astype(str)
        .str.strip()
    )

    results_df["method"] = (
        results_df["method"]
        .astype(str)
        .str.strip()
    )

    results_df["query"] = pd.to_numeric(
        results_df["query"],
        errors="coerce"
    )

    results_df["kappa"] = pd.to_numeric(
        results_df["kappa"],
        errors="coerce"
    )

    results_df = results_df.dropna(
        subset=[
            "query",
            "kappa"
        ]
    )

    # ========================================================
    # SVC
    # ========================================================

    results_df = results_df[
        results_df["classifier"]
        .str.upper()
        == "SVC"
    ].copy()

    # ========================================================
    # SOMENTE OS 26 MÉTODOS
    # ========================================================

    results_df = results_df[
        results_df["method"].isin(
            METHODS
        )
    ].copy()

    print()
    print(
        f"Datasets encontrados: "
        f"{results_df['dataset'].nunique()}"
    )

    print(
        f"Métodos encontrados: "
        f"{results_df['method'].nunique()}"
    )

    print(
        f"Queries encontradas: "
        f"{sorted(results_df['query'].astype(int).unique())}"
    )

    # ========================================================
    # MOSTRA QUAIS MÉTODOS EXISTEM
    # ========================================================

    found_methods = sorted(
        results_df["method"].unique()
    )

    missing_methods = sorted(
        set(METHODS) - set(found_methods)
    )

    print()
    print(
        f"Métodos presentes: {len(found_methods)}"
    )

    if missing_methods:

        print(
            "Métodos ausentes nos CSVs:"
        )

        for method in missing_methods:
            print(
                "  -",
                method
            )

    return results_df


# ============================================================
# REMOVER OS 10 PIORES (APENAS QUANDO HÁ >= 75 DATASETS)
# ============================================================

def remove_worst_datasets(
    results_df,
    initial_size
):
    # Nos cenarios filtrados (L0 = 100, 500, 1000), mantemos os datasets validos
    # ou removemos os 10 piores apenas no cenario completo (L0 = 5)
    if initial_size != 5 or results_df["dataset"].nunique() <= 15:
        return results_df.copy(), []

    datasets_to_rm = (

        results_df

        .groupby(
            [
                "dataset",
                "method",
                "fold"
            ]
        )

        .kappa

        .apply(
            np.trapz
        )

        .groupby(
            [
                "dataset",
                "method"
            ]
        )

        .mean()

        .groupby(
            "dataset"
        )

        .max()

        .sort_values()

        .head(10)

        .index
    )

    filtered = results_df[
        ~results_df["dataset"].isin(
            datasets_to_rm
        )
    ].copy()

    return filtered, list(
        datasets_to_rm
    )


# ============================================================
# RANKING
# ============================================================

def get_avg_ranks(
    results_df
):

    avg_kappa = (

        results_df

        .groupby(
            [
                "classifier",
                "method",
                "query",
                "dataset"
            ]
        )

        .kappa

        .mean()
    )

    ranks_df = (
        avg_kappa
        .to_frame(
            name="kappa"
        )
    )

    ranks_df["rank"] = (

        ranks_df

        .groupby(
            [
                "query",
                "dataset"
            ]
        )

        .kappa

        .rank(
            ascending=False
        )
    )

    avg_ranks = (

        ranks_df

        .groupby(
            [
                "classifier",
                "method",
                "query"
            ]
        )

        ["rank"]

        .mean()
    )

    return avg_ranks


# ============================================================
# GRÁFICO
# ============================================================

def plot_learning_curves(
    avg_ranks_df,
    batch_size,
    initial_size
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.grid(True)

    ax.invert_yaxis()

    # --------------------------------------------------------
    # EIXO X DINÂMICO (100 PARA L0=5, 50 PARA L0=100/500/1000)
    # --------------------------------------------------------

    max_q = MAX_QUERIES_BY_INIT.get(initial_size, 50)
    step_x = 10 if max_q == 100 else 5

    ax.set_xticks(
        np.arange(
            0,
            max_q + 1,
            step_x
        )
    )

    ax.set_xlim(
        0,
        int(max_q * 1.1)
    )

    ax.set_xlabel(
        "Queries ($i$)"
    )

    ax.set_ylabel(
        r"Mean. $\bar\kappa_i$ rank"
    )

    # --------------------------------------------------------
    # SVC
    # --------------------------------------------------------

    if "SVC" not in avg_ranks_df.index:

        print(
            "SVC não encontrado no ranking."
        )

        plt.close(fig)

        return None

    clf_ranks = avg_ranks_df.loc[
        "SVC"
    ]

    auc_scores = {}

    # --------------------------------------------------------
    # CURVAS
    # --------------------------------------------------------

    for method, data in clf_ranks.groupby(
        "method",
        sort=False
    ):

        data = data.sort_index(
            level="query"
        )

        temp = (
            data
            .reset_index(
                name="rank"
            )
            .sort_values(
                "query"
            )
        )

        window = SMOOTHING_WINDOW[
            batch_size
        ]

        smoothed = (

            temp["rank"]

            .rolling(
                window=window,
                min_periods=1
            )

            .mean()

            .to_numpy()
        )

        auc_scores[
            ABREV_DICT[method]
        ] = np.trapz(
            smoothed
        )

        queries = (
            temp["query"]
            .astype(int)
            .to_numpy()
        )

        x = (
            queries
            * batch_size
        )

        ax.plot(

            x,

            smoothed,

            get_style(
                method
            ),

            label=ABREV_DICT[
                method
            ],

            color=COLOR_DICT[
                method
            ],

            markevery=(
                0.3,
                0.2
            ),

            linewidth=(

                0.5

                if get_hm_type(
                    method
                )
                not in [
                    "classic",
                    "baseline"
                ]

                else 1.5
            )
        )

    # ========================================================
    # LEGENDA
    # ========================================================

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    label_to_handle = {
        label: handle

        for label, handle
        in zip(
            labels,
            handles
        )
    }

    sorted_labels = sorted(
        auc_scores,
        key=auc_scores.get
    )

    sorted_labels = [
        label
        for label in sorted_labels
        if label in label_to_handle
    ]

    ax.legend(

        [
            label_to_handle[
                label
            ]

            for label
            in sorted_labels
        ],

        sorted_labels,

        loc="upper right",

        bbox_to_anchor=(
            1.20,
            1.01
        ),

        ncols=1,

        prop={
            "size": 8
        },

        framealpha=1
    )

    return fig


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plota curvas de ranking para os cenários de L0 e batch sizes."
    )
    parser.add_argument(
        "--initial-sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        choices=[5, 100, 500, 1000],
        help="Quais tamanhos iniciais plotar (default: 100 500 1000)."
    )
    args = parser.parse_args()

    for initial_size in args.initial_sizes:

        plot_dir = get_plot_dir(initial_size)
        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for batch_size in BATCH_SIZES:

            results_df = load_results(
                batch_size,
                initial_size
            )

            if results_df is None or results_df.empty:
                continue

            results_df, removed = (
                remove_worst_datasets(
                    results_df,
                    initial_size
                )
            )

            if removed:
                print("\nDatasets removidos:")
                for dataset in removed:
                    print("  -", dataset)

            avg_ranks_df = (
                get_avg_ranks(
                    results_df
                )
            )

            fig = plot_learning_curves(
                avg_ranks_df,
                batch_size,
                initial_size
            )

            if fig is None:
                continue

            suffix = f"_init{initial_size}" if initial_size != 5 else ""
            output_base = (
                plot_dir
                / f"ranking_curve_SVC_batch{batch_size}{suffix}"
            )

            fig.savefig(
                f"{output_base}.png",
                dpi=300,
                bbox_inches="tight"
            )

            fig.savefig(
                f"{output_base}.eps",
                format="eps",
                bbox_inches="tight"
            )

            print()
            print("Salvo:")
            print(f"  {output_base}.png")
            print(f"  {output_base}.eps")

            plt.show()
            plt.close(fig)

    print()
    print("=" * 75)
    print("CONCLUÍDO")
    print("=" * 75)