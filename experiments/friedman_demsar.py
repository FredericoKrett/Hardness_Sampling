import argparse
import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, studentized_range

warnings.filterwarnings("ignore")

_DRIVE_ROOT = Path("/content/drive/MyDrive/hardness_sampling_hier_mcm")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    _DRIVE_ROOT
    if _DRIVE_ROOT.exists()
    else (SCRIPT_DIR.parent if SCRIPT_DIR.name == "experiments" else SCRIPT_DIR)
)
RESULTS_ROOT = REPO_ROOT / "results"
PLOTS_ROOT = REPO_ROOT / "friedman"

ABBR = {
    "k_disagreeing_neighbors_sampling": "kDN",
    "kdn_adj": r"$\mathrm{kDN}_{adj}$",
    "hier_kdn_sampling": "Hier+kDN",
    "hierarchical_sampling": "Hier",
    "mcm_sampling": "MCM",
    "ls_radius_sampling": "LSR",
    "local_set_cardinality_sampling": "LSCI",
    "class_likeliood_diff_sampling": "CLD",
    "class_likelihood_sampling": "CL",
    "tree_depth_unpruned_sampling": r"$\mathrm{TD}_u$",
    "tree_depth_pruned_sampling": r"$\mathrm{TD}_p$",
    "usefulness_sampling": "U",
    "harmfulness_sampling": "H",
    "random_sampling": "Rnd",
    "disjunct_class_percentage_sampling": "DCP",
    "disjunct_size_sampling": "DS",
    "borderline_points_sampling": "N1I",
    "intra_extra_ratio_sampling": "N2I",
    "margin_sampling": "MS",
    "least_confident_sampling": "LC",
    "entropy_sampling": "Ent",
    "f1_sampling": "F1I",
    "f2_sampling": "F2I",
    "f3_sampling": "F3I",
    "f4_sampling": "F4I",
    "expected_error_reduction": r"$\mathrm{EER}_{ent}$",
    "minority_value_sampling": "MV",
    "class_balance_sampling": "CB",
    "training_utility_sampling": "TU",
    "density_weighted_sampling": "ID",
}

TOP_METHODS_INIT = [
    "hier_kdn_sampling",
    "hierarchical_sampling",
    "k_disagreeing_neighbors_sampling",
    "random_sampling",
    "disjunct_size_sampling",
    "mcm_sampling",
    "local_set_cardinality_sampling",
    "harmfulness_sampling",
    "borderline_points_sampling",
    "class_likeliood_diff_sampling",
    "ls_radius_sampling",
    "kdn_adj",
]


def compute_cd_nemenyi(k: int, N: int, alpha: float = 0.05) -> float:
    q_alpha = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q_alpha * np.sqrt(k * (k + 1.0) / (6.0 * N)))


def find_maximal_cliques(mean_ranks: pd.Series, cd: float):
    ranks = mean_ranks.values
    k = len(ranks)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd + 1e-9:
            j += 1
        if j > i:
            is_sub = any(prev_i <= i and j <= prev_j for prev_i, prev_j in cliques)
            if not is_sub:
                cliques.append((i, j))
    return cliques


def load_observations(
    batch_size: int,
    initial_size: int = 5,
    classifier: str = "SVC",
    remove_n_worst: int = 10,
    by_fold: bool = True,
    max_queries: int = None,
):
    folder = (
        RESULTS_ROOT / f"svc_batch{batch_size}"
        if initial_size == 5
        else RESULTS_ROOT / f"init{initial_size}" / f"svc_batch{batch_size}"
    )

    if not folder.exists():
        print(f"Pasta não encontrada: {folder}")
        return pd.DataFrame(), [], 0

    all_csvs = sorted(glob.glob(str(folder / "*.csv")))
    svc_csvs = [
        f for f in all_csvs
        if f"_{classifier}_" in os.path.basename(f)
        or not any(c in os.path.basename(f) for c in ("_5NN_", "_DecisionTree_", "_GaussianNB_"))
    ]

    dfs = []
    for csv_path in svc_csvs:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            if "classifier" in df.columns:
                df = df[df["classifier"] == classifier]
            if not df.empty:
                dfs.append(df[["dataset", "method", "fold", "query", "kappa"]])
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(), [], 0

    results_df = pd.concat(dfs, ignore_index=True)
    if initial_size != 5:
        results_df = results_df[results_df["method"].isin(TOP_METHODS_INIT)].copy()

    if results_df.empty:
        return pd.DataFrame(), [], 0

    results_df = results_df.sort_values(["dataset", "method", "fold", "query"])

    if max_queries is not None and max_queries > 0:
        results_df["step"] = results_df.groupby(["dataset", "method", "fold"]).cumcount()
        max_steps = max(2, max_queries // batch_size)
        results_df = results_df[results_df["step"] <= max_steps]

    alp_D_df = (
        results_df.groupby(["dataset", "method", "fold"])["kappa"]
        .apply(np.trapezoid if hasattr(np, "trapezoid") else np.trapz)
        .rename("ALP")
        .reset_index()
    )

    total_methods = alp_D_df["method"].nunique()
    ds_method_counts = alp_D_df.groupby("dataset")["method"].nunique()
    valid_datasets = ds_method_counts[ds_method_counts == total_methods].index
    alp_D_df = alp_D_df[alp_D_df["dataset"].isin(valid_datasets)]

    removed_datasets = []
    if remove_n_worst > 0 and alp_D_df["dataset"].nunique() > remove_n_worst:
        datasets_to_rm = (
            alp_D_df.groupby(["dataset", "method"])["ALP"]
            .mean()
            .groupby("dataset")
            .max()
            .sort_values()
            .head(remove_n_worst)
            .index
        )
        removed_datasets = list(datasets_to_rm)
        alp_D_df = alp_D_df[~alp_D_df["dataset"].isin(datasets_to_rm)]

    n_datasets = alp_D_df["dataset"].nunique()

    if by_fold:
        observations_df = alp_D_df.pivot(
            index=["dataset", "fold"], columns="method", values="ALP"
        ).dropna(axis=0, how="any")
    else:
        observations_df = (
            alp_D_df.groupby(["dataset", "method"])["ALP"]
            .mean()
            .unstack("method")
            .dropna(axis=0, how="any")
        )

    return observations_df, removed_datasets, n_datasets


def draw_cd_diagram(
    ax,
    mean_ranks: pd.Series,
    cd: float,
    title: str = "",
):
    mean_ranks = mean_ranks.sort_values(ascending=True)
    k = len(mean_ranks)
    methods = list(mean_ranks.index)
    ranks = mean_ranks.values

    n_right = math.ceil(k / 2.0)
    n_left = k - n_right

    cliques = find_maximal_cliques(mean_ranks, cd)
    n_cliques = len(cliques)

    y_axis = 1.0
    y_cd = 1.45
    clique_start_y = 0.68
    clique_step_y = 0.22
    label_start_y = clique_start_y - max(1, n_cliques) * clique_step_y - 0.25
    label_step_y = 0.36

    n_rows = max(n_right, n_left)
    y_min = label_start_y - n_rows * label_step_y - 0.4
    y_max = 2.05

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")

    x0, x1 = 0.22, 0.78

    def rank_to_x(r):
        return x0 + (k - r) / max(1.0, (k - 1.0)) * (x1 - x0)

    ax.plot([x0, x1], [y_axis, y_axis], color="black", lw=1.2)

    tick_step = 1 if k <= 30 else 2
    ticks = sorted(set(list(range(1, k + 1, tick_step)) + [1, k]))
    for t in ticks:
        xt = rank_to_x(t)
        ax.plot([xt, xt], [y_axis, y_axis + 0.08], color="black", lw=1.0)
        ax.text(xt, y_axis + 0.13, f"{t}", ha="center", va="bottom", fontsize=8.5, color="black")

    cd_width = (cd / max(1.0, (k - 1.0))) * (x1 - x0)
    x_cd_start = x0
    x_cd_end = min(x1, x0 + cd_width)
    ax.plot([x_cd_start, x_cd_end], [y_cd, y_cd], color="black", lw=1.8)
    ax.plot([x_cd_start, x_cd_start], [y_cd - 0.06, y_cd + 0.06], color="black", lw=1.4)
    ax.plot([x_cd_end, x_cd_end], [y_cd - 0.06, y_cd + 0.06], color="black", lw=1.4)
    ax.text(
        (x_cd_start + x_cd_end) / 2.0,
        y_cd + 0.10,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    if title:
        ax.text(
            0.5,
            y_max - 0.05,
            title,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    for idx_c, (i, j) in enumerate(cliques):
        yc = clique_start_y - idx_c * clique_step_y
        xi = rank_to_x(ranks[i])
        xj = rank_to_x(ranks[j])
        ax.plot([xj, xi], [yc, yc], color="black", lw=3.0, solid_capstyle="butt")

    for idx in range(n_right):
        m = methods[idx]
        r = ranks[idx]
        xm = rank_to_x(r)
        ym = label_start_y - idx * label_step_y
        label = ABBR.get(m, m)

        ax.plot([xm, xm], [y_axis, ym], color="black", lw=0.7, ls="-")
        ax.plot([xm, x1 + 0.03], [ym, ym], color="black", lw=0.7, ls="-")
        ax.plot(xm, y_axis, "o", color="black", markersize=3.5)
        ax.text(
            x1 + 0.04,
            ym,
            f"{r:.2f}  {label}",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
        )

    for idx_l in range(n_left):
        idx = k - 1 - idx_l
        m = methods[idx]
        r = ranks[idx]
        xm = rank_to_x(r)
        ym = label_start_y - idx_l * label_step_y
        label = ABBR.get(m, m)

        ax.plot([xm, xm], [y_axis, ym], color="black", lw=0.7, ls="-")
        ax.plot([x0 - 0.03, xm], [ym, ym], color="black", lw=0.7, ls="-")
        ax.plot(xm, y_axis, "o", color="black", markersize=3.5)
        ax.text(
            x0 - 0.04,
            ym,
            f"{label}  {r:.2f}",
            ha="right",
            va="center",
            fontsize=9,
            color="black",
        )


def run_friedman_analysis(
    batches=(1, 5, 10, 25),
    initial_size=5,
    classifier="SVC",
    remove_n_worst=10,
    by_fold=True,
    max_queries=None,
    alpha=0.05,
):
    eff_rm = remove_n_worst if initial_size == 5 else 0
    mode_tag = "byfold" if by_fold else "bydataset"
    suffix = f"bracis_{classifier}_{mode_tag}_init{initial_size}_rm{eff_rm}"
    out_dir = PLOTS_ROOT / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ranks = {}
    fig_grid, axes_grid = plt.subplots(2, 2, figsize=(18, 15), dpi=300)
    axes_flat = axes_grid.flatten()

    for idx_b, b in enumerate(batches):
        obs_df, removed, n_ds = load_observations(
            batch_size=b,
            initial_size=initial_size,
            classifier=classifier,
            remove_n_worst=eff_rm,
            by_fold=by_fold,
            max_queries=max_queries,
        )
        if obs_df.empty:
            print(f"[L0={initial_size} | Batch {b}] Sem dados suficientes.")
            continue

        N_obs, k = obs_df.shape
        ranks_df = obs_df.rank(axis=1, ascending=False, method="average")
        mean_ranks = ranks_df.mean(axis=0).sort_values(ascending=True)
        all_ranks[f"batch_{b}"] = mean_ranks

        stat, p_val = friedmanchisquare(*[obs_df[col].values for col in obs_df.columns])
        cd = compute_cd_nemenyi(k=k, N=N_obs, alpha=alpha)

        print("=" * 75)
        print(
            f"BATCH {b} ({classifier}) | L0={initial_size} | Datasets={n_ds} (N_obs={N_obs}) | k={k} métodos"
        )
        print(f"Friedman chi2 = {stat:.2f} | p-value = {p_val:.2e} | CD = {cd:.2f}")
        print("-" * 75)
        for pos, (m, r_val) in enumerate(mean_ranks.head(12).items(), 1):
            print(f"  {pos:2d}. {ABBR.get(m, m):10s} ({m:35s}) -> Rank Médio: {r_val:.2f}")

        fig_single, ax_single = plt.subplots(figsize=(10, 7.5), dpi=300)
        title_str = f"Batch {b} - {classifier} (L0={initial_size} | {n_ds} datasets, N={N_obs} | CD={cd:.2f})"
        draw_cd_diagram(ax_single, mean_ranks, cd, title=title_str)
        plt.tight_layout()
        fig_single.savefig(out_dir / f"friedman_cd_batch{b}.png", dpi=300, bbox_inches="tight")
        fig_single.savefig(out_dir / f"friedman_cd_batch{b}.eps", format="eps", bbox_inches="tight")
        plt.close(fig_single)

        if idx_b < 4:
            draw_cd_diagram(axes_flat[idx_b], mean_ranks, cd, title=title_str)

    if all_ranks:
        plt.tight_layout()
        fig_grid.savefig(out_dir / f"friedman_cd_all_batches_{suffix}.png", dpi=300, bbox_inches="tight")
        fig_grid.savefig(out_dir / f"friedman_cd_all_batches_{suffix}.eps", format="eps", bbox_inches="tight")
        plt.close(fig_grid)

        ranks_table = pd.DataFrame(all_ranks)
        ranks_table["mean_across_batches"] = ranks_table.mean(axis=1)
        ranks_table = ranks_table.sort_values("mean_across_batches", ascending=True)
        ranks_table.to_csv(out_dir / f"friedman_ranks_summary_{suffix}.csv")

        print("\n" + "=" * 75)
        print(f"RANKING GLOBAL CONSOLIDADO (L0={initial_size} | {classifier}):")
        print("=" * 75)
        for pos, (m, row) in enumerate(ranks_table.iterrows(), 1):
            b_str = " | ".join(
                [f"B{b}: {row[f'batch_{b}']:.2f}" for b in batches if f"batch_{b}" in row]
            )
            print(
                f"  {pos:2d}. {ABBR.get(m, m):10s} ({m:35s}) -> Média: {row['mean_across_batches']:.2f}  [{b_str}]"
            )

        print(f"\nGráficos salvos em: {out_dir}")
    else:
        plt.close(fig_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-size", "--initial-sizes",
        dest="initial_sizes",
        type=int,
        nargs="+",
        default=[5],
        choices=[5, 100, 500, 1000],
    )
    parser.add_argument("--classifier", type=str, default="SVC")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 5, 10, 25])
    parser.add_argument("--remove-worst", type=int, default=10)
    parser.add_argument("--by-dataset", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    for init_sz in args.initial_sizes:
        run_friedman_analysis(
            batches=args.batches,
            initial_size=init_sz,
            classifier=args.classifier,
            remove_n_worst=args.remove_worst,
            by_fold=not args.by_dataset,
            max_queries=args.max_queries,
        )