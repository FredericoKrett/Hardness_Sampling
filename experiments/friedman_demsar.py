import argparse
import glob
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_DRIVE_ROOT = Path("/content/drive/MyDrive/hardness_sampling_hier_mcm")
REPO_ROOT = _DRIVE_ROOT if _DRIVE_ROOT.exists() else Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "friedman"

ABBR = {
    "k_disagreeing_neighbors_sampling": "kDN",
    "kdn_adj": r"$\mathrm{kDN}_{adj}$",
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


def load_clean_auc_matrix(batch_size, initial_size=5):
    folder = (
        RESULTS_ROOT / f"svc_batch{batch_size}"
        if initial_size == 5
        else RESULTS_ROOT / f"init{initial_size}" / f"svc_batch{batch_size}"
    )

    min_steps = max(2, min(10, 50 // batch_size))

    records = []
    for csv_path in glob.glob(str(folder / "*.csv")):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=["dataset", "method", "run", "fold", "query", "kappa"]
            )
            if df.empty:
                continue
            curve = (
                df.groupby(["dataset", "method", "query"])["kappa"]
                .mean()
                .reset_index()
            )
            records.append(curve)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    full_curves = pd.concat(records, ignore_index=True)
    total_methods = full_curves["method"].nunique()

    auc_rows = []
    for ds, df_ds in full_curves.groupby("dataset"):
        if df_ds["method"].nunique() < total_methods:
            continue

        query_counts = df_ds.groupby("query")["method"].nunique()
        common_queries = query_counts[query_counts == total_methods].index

        if len(common_queries) < min_steps:
            continue

        df_common = df_ds[df_ds["query"].isin(common_queries)].sort_values("query")
        for m, df_m in df_common.groupby("method"):
            y = df_m["kappa"].values
            if len(y) > 1:
                auc_val = (
                    np.trapezoid(y) / (len(y) - 1)
                    if hasattr(np, "trapezoid")
                    else np.trapz(y) / (len(y) - 1)
                )
            else:
                auc_val = float(y[0])
            auc_rows.append({"dataset": ds, "method": m, "auc": auc_val})

    if not auc_rows:
        return pd.DataFrame()

    auc_df = pd.DataFrame(auc_rows)
    pivot = auc_df.pivot_table(index="dataset", columns="method", values="auc")
    return pivot.dropna(axis=0, how="any")


def find_maximal_cliques(ranks, cd):
    k = len(ranks)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        if j > i:
            if not cliques or j > cliques[-1][1]:
                cliques.append((i, j))
    return cliques


def plot_demsar_cd_diagram(avg_ranks, cd, out_path, title=None):
    methods_sorted = list(avg_ranks.index)
    ranks_sorted = avg_ranks.values
    k = len(methods_sorted)

    n_right = (k + 1) // 2
    right_methods = methods_sorted[:n_right]
    right_ranks = ranks_sorted[:n_right]

    left_methods = methods_sorted[n_right:][::-1]
    left_ranks = ranks_sorted[n_right:][::-1]

    cliques = find_maximal_cliques(ranks_sorted, cd)
    n_cliques = len(cliques)

    max_rows = max(len(right_methods), len(left_methods))
    clique_space = max(1.2, n_cliques * 0.38 + 0.6)
    y_axis = max_rows * 0.55 + clique_space + 1.0
    fig_height = max(5.5, y_axis + 1.4)

    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=300)

    def r2x(r):
        return k + 1 - r

    x_left_edge = 1.0
    x_right_edge = float(k)

    ax.plot([x_left_edge, x_right_edge], [y_axis, y_axis], color="black", lw=1.2)

    for r in range(1, k + 1):
        x = r2x(r)
        ax.plot([x, x], [y_axis, y_axis + 0.18], color="black", lw=1.0)
        ax.text(x, y_axis + 0.28, f"{r}", ha="center", va="bottom", fontsize=10)
        if r < k:
            x_half = r2x(r + 0.5)
            ax.plot([x_half, x_half], [y_axis, y_axis + 0.09], color="black", lw=0.7)

    y_cd = y_axis + 0.95
    x_cd_start = x_left_edge
    x_cd_end = x_left_edge + cd
    ax.plot([x_cd_start, x_cd_end], [y_cd, y_cd], color="black", lw=1.2)
    ax.plot([x_cd_start, x_cd_start], [y_cd - 0.15, y_cd + 0.15], color="black", lw=1.2)
    ax.plot([x_cd_end, x_cd_end], [y_cd - 0.15, y_cd + 0.15], color="black", lw=1.2)
    ax.text((x_cd_start + x_cd_end) / 2, y_cd + 0.18, "CD", ha="center", va="bottom", fontsize=11)

    y_first_clique = y_axis - 0.45
    for idx_c, (i, j) in enumerate(cliques):
        y_c = y_first_clique - idx_c * 0.34
        x1 = r2x(ranks_sorted[j]) - 0.08
        x2 = r2x(ranks_sorted[i]) + 0.08
        ax.plot([x1, x2], [y_c, y_c], color="black", lw=3.2, solid_capstyle="butt")

    y_top_label = y_axis - clique_space - 0.3
    row_step = 0.52

    x_label_right = x_right_edge + 0.6
    for idx, (m, r) in enumerate(zip(right_methods, right_ranks)):
        y_row = y_top_label - idx * row_step
        x_r = r2x(r)
        label = ABBR.get(m, m)
        ax.plot([x_r, x_r, x_label_right], [y_axis, y_row, y_row], color="black", lw=0.9)
        ax.text(x_label_right + 0.25, y_row, label, ha="left", va="center", fontsize=11)

    x_label_left = x_left_edge - 0.6
    for idx, (m, r) in enumerate(zip(left_methods, left_ranks)):
        y_row = y_top_label - idx * row_step
        x_r = r2x(r)
        label = ABBR.get(m, m)
        ax.plot([x_r, x_r, x_label_left], [y_axis, y_row, y_row], color="black", lw=0.9)
        ax.text(x_label_left - 0.25, y_row, label, ha="right", va="center", fontsize=11)

    if title:
        ax.set_title(title, fontsize=11, pad=15)

    ax.set_xlim(x_left_edge - 4.5, x_right_edge + 4.5)
    ax.set_ylim(y_top_label - max_rows * row_step - 0.4, y_cd + 0.7)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(initial_size=5):
    out_folder = OUT_DIR if initial_size == 5 else OUT_DIR / f"init{initial_size}"
    out_folder.mkdir(parents=True, exist_ok=True)

    all_ranks = []
    for b in [1, 5, 10, 25]:
        pivot = load_clean_auc_matrix(b, initial_size=initial_size)
        if pivot.empty:
            print(f"[AVISO] Sem datasets completos para L0={initial_size} | Batch {b}")
            continue

        n_ds, k_m = pivot.shape
        stat, p_val = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        ranks_df = pivot.rank(axis=1, ascending=False, method="average")
        avg_ranks = ranks_df.mean(axis=0).sort_values(ascending=True)

        q_alpha = studentized_range.ppf(1 - 0.05, k_m, np.inf) / np.sqrt(2)
        cd = q_alpha * np.sqrt(k_m * (k_m + 1) / (6.0 * n_ds))

        all_ranks.append(avg_ranks.rename(f"batch_{b}"))

        img_path = out_folder / f"demsar_cd_L0_{initial_size}_batch{b}.png"
        plot_demsar_cd_diagram(
            avg_ranks,
            cd,
            img_path,
            title=f"Batch {b} (L0={initial_size} | N={n_ds} datasets | CD={cd:.2f} | p={p_val:.1e})",
        )
        print(f"Salvo: {img_path} (N={n_ds} datasets | CD={cd:.2f})")

    if all_ranks:
        summary_df = pd.concat(all_ranks, axis=1)
        summary_df["mean_rank"] = summary_df.mean(axis=1)
        summary_df["abbr"] = [ABBR.get(m, m) for m in summary_df.index]
        summary_df = summary_df.sort_values("mean_rank")
        summary_df.to_csv(out_folder / f"ranking_consolidado_L0_{initial_size}.csv")
        print("\nTOP 12 CONSOLIDADO:")
        print(summary_df.head(12).round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-size", type=int, default=5)
    args = parser.parse_args()
    main(args.initial_size)