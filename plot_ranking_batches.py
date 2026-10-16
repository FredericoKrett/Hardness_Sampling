import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

warnings.filterwarnings("ignore")

REPO_ROOT = Path(
    "/content/drive/MyDrive/hardness_sampling_svc"
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


# ============================================================
# 24 MÉTODOS
# ============================================================

ABREV_DICT = {

    "random_sampling": "Rnd",

    "margin_sampling": "MS",

    "training_utility_sampling": "TU",

    "density_weighted_sampling": "ID",

    "expected_error_reduction": r"EER$_{ent}$",

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

    "minority_value_sampling": "MV"

    "expected_error_reduction": r"EER$_{ent}$",

    "hierarchical_sampling": "Hier",

    "mcm_sampling": "MCM",
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
# CARREGAR CSVs
# ============================================================

def load_results(batch_size):

    results_dir = (
        RESULTS_ROOT
        / f"svc_batch{batch_size}"
    )

    print()
    print("=" * 75)
    print(f"BATCH {batch_size}")
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

    # Usa case-insensitive para evitar o problema
    # de "SVC", "svc", etc.
    results_df = results_df[
        results_df["classifier"]
        .str.upper()
        == "SVC"
    ].copy()

    # ========================================================
    # SOMENTE OS 24 MÉTODOS
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
# REMOVER OS 10 PIORES
# ============================================================

def remove_worst_datasets(
    results_df
):

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

    # --------------------------------------------------------
    # Média de Kappa por:
    #
    # classifier / method / query / dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Rank:
    #
    # maior Kappa = rank 1
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Rank médio entre datasets
    # --------------------------------------------------------

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
    batch_size
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.grid(True)

    ax.invert_yaxis()

    # --------------------------------------------------------
    # EIXO X
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            0,
            101,
            10
        )
    )

    ax.set_xlim(
        0,
        110
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

        # Ordena query
        data = data.sort_index(
            level="query"
        )

        # ----------------------------------------------------
        # Transforma em DataFrame
        # ----------------------------------------------------

        temp = (
            data
            .reset_index(
                name="rank"
            )
            .sort_values(
                "query"
            )
        )

        # ----------------------------------------------------
        # Média móvel
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # AUC para ordenar a legenda
        # ----------------------------------------------------

        auc_scores[
            ABREV_DICT[method]
        ] = np.trapz(
            smoothed
        )

        # ----------------------------------------------------
        # QUERY -> INSTÂNCIAS CONSULTADAS
        # ----------------------------------------------------

        queries = (
            temp["query"]
            .astype(int)
            .to_numpy()
        )

        x = (
            queries
            * batch_size
        )

        # ----------------------------------------------------
        # PLOT
        # ----------------------------------------------------

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

    # Só coloca na legenda o que realmente foi plotado.
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
# EXECUTAR 1, 5, 10 E 25
# ============================================================

PLOT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


for batch_size in BATCH_SIZES:

    results_df = load_results(
        batch_size
    )

    if results_df is None:
        continue

    if results_df.empty:

        print(
            f"\n[ERRO] Batch {batch_size}: "
            "nenhum dado SVC encontrado."
        )

        continue

    # ========================================================
    # REMOVER OS 10 PIORES
    # ========================================================

    results_df, removed = (
        remove_worst_datasets(
            results_df
        )
    )

    print()
    print(
        "Datasets removidos:"
    )

    for dataset in removed:

        print(
            "  -",
            dataset
        )

    # ========================================================
    # RANKING
    # ========================================================

    avg_ranks_df = (
        get_avg_ranks(
            results_df
        )
    )

    # ========================================================
    # GRÁFICO
    # ========================================================

    fig = plot_learning_curves(
        avg_ranks_df,
        batch_size
    )

    if fig is None:
        continue

    # ========================================================
    # SALVAR
    # ========================================================

    output_base = (
        PLOT_ROOT
        / f"ranking_curve_SVC_batch{batch_size}"
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
    print(
        f"Salvo:"
    )

    print(
        f"  {output_base}.png"
    )

    print(
        f"  {output_base}.eps"
    )

    plt.show()

    plt.close(fig)


print()
print("=" * 75)
print("CONCLUÍDO")
print("=" * 75)
print()
print(
    f"Gráficos em:\n{PLOT_ROOT}"
)