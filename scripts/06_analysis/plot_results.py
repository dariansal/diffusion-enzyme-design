#!/usr/bin/env python3
"""
plot_results.py

Generates all publication-quality figures from the computed metrics:

Figure 1: Catalytic RMSD by conditioning regime (box + strip plots)
Figure 2: Local pLDDT at catalytic residues by regime
Figure 3: Structural variance (inter-model RMSD) by regime
Figure 4: RMSD vs pLDDT scatter coloured by regime
Figure 5: Combined panel for the two enzyme systems

All figures saved as both PDF (for publication) and PNG (for reports).

Usage:
  conda run -p /hpc/group/naderilab/darian/conda_environments/enz_analysis \
      python scripts/06_analysis/plot_results.py
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT = Path("/hpc/group/naderilab/darian/Enz")
ANA_DIR = PROJECT / "results" / "analysis"
FIG_DIR = ANA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette for regimes
REGIME_COLORS = {
    "motif_only": "#4878CF",   # blue
    "shell5":     "#6ACC65",   # green
    "shell8":     "#D65F5F",   # red
}
REGIME_LABELS = {
    "motif_only": "Motif only\n(3 residues fixed)",
    "shell5":     "5 Å shell\n(~15 residues fixed)",
    "shell8":     "8 Å shell\n(~30 residues fixed)",
}
REGIME_ORDER = ["motif_only", "shell5", "shell8"]

ENZYME_LABELS = {
    "1ppf": "Subtilisin BPN'\n(Ser-His-Asp triad)",
    "1ca2": "Carbonic Anhydrase II\n(Zn²⁺-binding His triad)",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(ANA_DIR / "metrics.csv")
    variance = pd.read_csv(ANA_DIR / "variance.csv")
    summary = pd.read_csv(ANA_DIR / "summary.csv")
    log.info(f"Loaded {len(metrics)} predictions, {len(variance)} sequences")
    return metrics, variance, summary


# ---------------------------------------------------------------------------
# Helper: jittered strip plot overlay
# ---------------------------------------------------------------------------

def strip_plot(ax, data_by_group, positions, colors, alpha=0.5, jitter=0.08):
    """Overlay individual data points on a box plot."""
    rng = np.random.default_rng(42)
    for pos, (group, color) in zip(positions, zip(data_by_group, colors)):
        vals = np.array(group)
        x = pos + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(x, vals, color=color, alpha=alpha, s=20, zorder=3,
                   linewidths=0)


# ---------------------------------------------------------------------------
# Figure 1 & 2: RMSD and pLDDT by regime, split by enzyme
# ---------------------------------------------------------------------------

def plot_metric_by_regime(
    metrics: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    fig_name: str,
    invert_better: bool = True,  # True = lower is better
):
    """Box + strip plot for one metric, two enzymes side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for ax, (enz_id, enz_label) in zip(axes, ENZYME_LABELS.items()):
        sub = metrics[metrics["enzyme_id"] == enz_id].copy()
        sub = sub[sub["regime"].isin(REGIME_ORDER)]
        sub = sub.dropna(subset=[metric_col])

        data_groups = [
            sub[sub["regime"] == r][metric_col].values
            for r in REGIME_ORDER
        ]
        positions = list(range(len(REGIME_ORDER)))
        colors = [REGIME_COLORS[r] for r in REGIME_ORDER]

        bp = ax.boxplot(
            data_groups,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            flierprops={"marker": "", "markersize": 0},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        strip_plot(ax, data_groups, positions, colors)

        # Mean markers
        for pos, grp in zip(positions, data_groups):
            if len(grp) > 0:
                ax.scatter(pos, np.mean(grp), marker="D", color="white",
                           edgecolors="black", s=40, zorder=5, linewidths=1.2)

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [REGIME_LABELS[r] for r in REGIME_ORDER],
            fontsize=9, ha="center"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(enz_label, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate n
        for pos, grp in zip(positions, data_groups):
            ax.annotate(f"n={len(grp)}", xy=(pos, ax.get_ylim()[0]),
                        ha="center", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    _save_fig(fig, fig_name)


# ---------------------------------------------------------------------------
# Figure 3: Structural variance
# ---------------------------------------------------------------------------

def plot_structural_variance(variance: pd.DataFrame):
    """Plot inter-model structural variance per regime."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    fig.suptitle(
        "Structural Variance Across AlphaFold2 Models\n"
        "(Pairwise Catalytic RMSD Among 5 Model Predictions)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, (enz_id, enz_label) in zip(axes, ENZYME_LABELS.items()):
        sub = variance[variance["enzyme_id"] == enz_id].copy()
        sub = sub[sub["regime"].isin(REGIME_ORDER)]
        sub = sub.dropna(subset=["structural_variance"])

        data_groups = [
            sub[sub["regime"] == r]["structural_variance"].values
            for r in REGIME_ORDER
        ]
        positions = list(range(len(REGIME_ORDER)))
        colors = [REGIME_COLORS[r] for r in REGIME_ORDER]

        bp = ax.boxplot(
            data_groups, positions=positions, widths=0.5,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            flierprops={"marker": ""},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        strip_plot(ax, data_groups, positions, colors)

        for pos, grp in zip(positions, data_groups):
            if len(grp) > 0:
                ax.scatter(pos, np.mean(grp), marker="D", color="white",
                           edgecolors="black", s=40, zorder=5, linewidths=1.2)

        ax.set_xticks(positions)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=9)
        ax.set_ylabel("Structural Variance (Å RMSD)")
        ax.set_title(enz_label, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, "fig3_structural_variance")


# ---------------------------------------------------------------------------
# Figure 4: RMSD vs pLDDT scatter
# ---------------------------------------------------------------------------

def plot_rmsd_vs_plddt(metrics: pd.DataFrame):
    """Scatter plot of catalytic RMSD vs catalytic pLDDT, coloured by regime."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Catalytic RMSD vs Local pLDDT\n(ideal designs: low RMSD, high pLDDT)",
        fontsize=13, fontweight="bold", y=1.01
    )

    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], label=r.replace("_", " "), alpha=0.8)
        for r in REGIME_ORDER
    ]

    for ax, (enz_id, enz_label) in zip(axes, ENZYME_LABELS.items()):
        sub = metrics[metrics["enzyme_id"] == enz_id].copy()
        sub = sub[sub["regime"].isin(REGIME_ORDER)]
        sub = sub.dropna(subset=["catalytic_rmsd", "plddt_catalytic"])

        for regime in REGIME_ORDER:
            grp = sub[sub["regime"] == regime]
            ax.scatter(
                grp["catalytic_rmsd"],
                grp["plddt_catalytic"],
                color=REGIME_COLORS[regime],
                alpha=0.6, s=40, zorder=3,
                label=regime.replace("_", " "),
            )

        ax.set_xlabel("Catalytic RMSD (Å)")
        ax.set_ylabel("Local pLDDT at Catalytic Residues")
        ax.set_title(enz_label, fontsize=11)
        ax.axhline(70, color="gray", linestyle="--", lw=1, label="pLDDT=70 threshold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].legend(handles=legend_patches, title="Conditioning regime",
                   bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    _save_fig(fig, "fig4_rmsd_vs_plddt")


# ---------------------------------------------------------------------------
# Figure 5: Combined summary panel
# ---------------------------------------------------------------------------

def plot_summary_panel(summary: pd.DataFrame):
    """
    Publication-ready 3×2 panel showing all three metrics for both enzymes.
    Rows: Catalytic RMSD, Local pLDDT, Structural Variance
    Cols: 1PPF, 1CA2
    """
    metrics_to_plot = [
        ("mean_cat_rmsd",            "std_cat_rmsd",           "Catalytic RMSD (Å)",          True),
        ("mean_plddt_cat",           "std_plddt_cat",          "Local pLDDT (catalytic)",      False),
        ("mean_structural_variance", "std_structural_variance", "Structural Variance (Å RMSD)", True),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        "Effect of Conditioning Strength on Active-Site Geometric Precision",
        fontsize=14, fontweight="bold", y=1.01
    )

    regime_x = {r: i for i, r in enumerate(REGIME_ORDER)}
    x_ticks = list(regime_x.values())
    x_labels = ["Motif only", "5 Å shell", "8 Å shell"]

    for row, (mean_col, std_col, ylabel, lower_is_better) in enumerate(metrics_to_plot):
        if mean_col not in summary.columns:
            log.warning(f"Column '{mean_col}' not in summary — skipping")
            continue
        for col, (enz_id, enz_label) in enumerate(ENZYME_LABELS.items()):
            ax = axes[row][col]
            sub = summary[summary["enzyme_id"] == enz_id].copy()
            sub = sub[sub["regime"].isin(REGIME_ORDER)]

            xs, ys, errs = [], [], []
            colors = []
            for regime in REGIME_ORDER:
                row_data = sub[sub["regime"] == regime]
                if row_data.empty:
                    continue
                xs.append(regime_x[regime])
                ys.append(row_data[mean_col].values[0])
                err = row_data[std_col].values[0] if std_col and std_col in sub.columns else 0
                errs.append(err)
                colors.append(REGIME_COLORS[regime])

            bars = ax.bar(xs, ys, color=colors, alpha=0.8, width=0.6, zorder=3)
            if any(e > 0 for e in errs):
                ax.errorbar(xs, ys, yerr=errs, fmt="none", color="black",
                            capsize=4, linewidth=1.5, zorder=4)

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(enz_label, fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Direction-of-better annotation
            direction = "↓ better" if lower_is_better else "↑ better"
            ax.annotate(direction, xy=(0.97, 0.97),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=9, color="gray")

    plt.tight_layout()
    _save_fig(fig, "fig5_summary_panel")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str):
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        log.info(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Generating figures ===")

    metrics, variance, summary = load_data()

    log.info("\n[Figure 1] Catalytic RMSD by regime")
    plot_metric_by_regime(
        metrics, "catalytic_rmsd",
        ylabel="Catalytic RMSD (Å)",
        title="Catalytic Active-Site RMSD vs Conditioning Strength\n"
              "(lower = better geometric precision)",
        fig_name="fig1_catalytic_rmsd",
    )

    log.info("\n[Figure 2] Local pLDDT by regime")
    plot_metric_by_regime(
        metrics, "plddt_catalytic",
        ylabel="Local pLDDT at Catalytic Residues",
        title="AlphaFold2 Confidence at Catalytic Residues vs Conditioning Strength\n"
              "(higher = more confident local structure)",
        fig_name="fig2_local_plddt",
        invert_better=False,
    )

    log.info("\n[Figure 3] Structural variance")
    plot_structural_variance(variance)

    log.info("\n[Figure 4] RMSD vs pLDDT scatter")
    plot_rmsd_vs_plddt(metrics)

    log.info("\n[Figure 5] Combined summary panel")
    plot_summary_panel(summary)

    log.info(f"\nAll figures saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()
