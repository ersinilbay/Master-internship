import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib.colors import Normalize as _Normalize, TwoSlopeNorm as _TwoSlopeNorm
from scipy import sparse
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from config import (
    PAPER_CMAP,
    STATE_CATEGORIES,
    SET1_COLORS,
    TRANSITION_CMAP,
    UMAP_LABEL_FS,
)


def add_corr_box(ax, x, y, method="pearson", loc="br", show_n=False):
    """Draw a small correlation badge on an axis."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return

    if method == "spearman":
        r, _ = spearmanr(x[m], y[m])
        text = f"Spearman ρ = {r:.2f}"
    else:
        r, _ = pearsonr(x[m], y[m])
        text = f"Pearson r = {r:.2f}"

    if show_n:
        text += f"\n n = {int(m.sum())}"

    pos = {
        "tl": (0.02, 0.98, dict(ha="left", va="top")),
        "tr": (0.98, 0.98, dict(ha="right", va="top")),
        "bl": (0.02, 0.02, dict(ha="left", va="bottom")),
        "br": (0.95, 0.03, dict(ha="right", va="bottom")),
    }[loc]

    ax.text(
        pos[0],
        pos[1],
        text,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="square,pad=0.18",
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.8,
            alpha=0.9,
        ),
        **pos[2],
    )


def plot_umap_scalar(ax, X_umap, vals, vmin=None, vmax=None, s=6, alpha=0.9):
    norm = _Normalize(vmin=vmin, vmax=vmax)
    sca = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=vals,
        s=s,
        alpha=alpha,
        cmap=PAPER_CMAP,
        norm=norm,
        linewidths=0,
    )
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
    ax.set_aspect("equal", adjustable="box")
    return sca


def plot_umap_transition(ax, X_umap, vals, s=6, alpha=0.9):
    lo, hi = np.nanpercentile(vals, [1, 99])
    m = float(max(abs(lo), abs(hi)))
    norm = _TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=+m)
    sca = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=vals,
        s=s,
        alpha=alpha,
        cmap=TRANSITION_CMAP,
        norm=norm,
        linewidths=0,
    )
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
    ax.set_aspect("equal", adjustable="box")
    return sca


def add_cut_lines_minimal():
    """Just red dashed cut lines + small red labels, same as original."""
    axes = plt.gcf().axes
    cut = "#B22222"
    dash = (0, (7, 4))
    cuts = [
        [("≥ 200", 200), ("≤ 2500", 2500)],
        [("≤ 5000", 5000)],
        [("≤ 5", 5)],
    ]
    for ax, arr in zip(axes, cuts):
        for txt, y in arr:
            ax.axhline(y, color=cut, lw=2.4, ls=dash, zorder=10)
            ax.text(
                0.01,
                y,
                txt,
                transform=ax.get_yaxis_transform(),
                color=cut,
                fontsize=9,
                ha="left",
                va="bottom",
                zorder=11,
            )


def plot_qc_violins(adata, out, filename, pre_or_post_label="POST"):
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        log=False,
        multi_panel=True,
        show=False,
    )

    labels = [
        "Genes detected (UMIs)",
        "Total UMIs",
        "Mitochondrial UMIs (%)",
    ]
    for ax, lab in zip(plt.gcf().axes, labels):
        ax.set_ylabel(lab)

    add_cut_lines_minimal()
    plt.tight_layout()
    out.savefig_named(filename, format="svg")
    plt.show()


def plot_qc_scatter(adata, out, filename="scatter_plots.svg"):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.5))

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        ax=axes[0],
        title="Genes vs total UMIs",
        size=20,
        show=False,
    )
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        ax=axes[1],
        title="% MT vs total UMIs",
        size=20,
        show=False,
    )

    axes[0].set_xlabel("Total UMIs")
    axes[0].set_ylabel("Genes detected (UMIs)")
    axes[1].set_xlabel("Total UMIs")
    axes[1].set_ylabel("Mitochondrial UMIs (%)")

    for ax in axes:
        ax.set_box_aspect(1.0)

    add_corr_box(
        axes[0],
        adata.obs["total_counts"].values,
        adata.obs["n_genes_by_counts"].values,
        loc="br",
    )
    add_corr_box(
        axes[1],
        adata.obs["total_counts"].values,
        adata.obs["pct_counts_mt"].values,
        loc="br",
    )

    plt.tight_layout()
    out.savefig_named(filename, format="svg")
    plt.show()


def plot_scalar_umaps(adata_umap, X_umap, xlim_glob, ylim_glob, out):
    mean_ntr_vals = adata_umap.obs["mean_ntr"].to_numpy()
    total_counts_vals = adata_umap.obs["total_counts"].to_numpy()
    pct_mt_vals = adata_umap.obs["pct_counts_mt"].to_numpy()

    ntr_lo, ntr_hi = np.nanpercentile(mean_ntr_vals, [1, 99])
    tot_lo, tot_hi = np.nanpercentile(total_counts_vals, [1, 99])
    mt_lo, mt_hi = np.nanpercentile(pct_mt_vals, [1, 99])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.6, 2.2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    sc1 = plot_umap_scalar(axes[0], X_umap, mean_ntr_vals, vmin=ntr_lo, vmax=ntr_hi)
    axes[0].set_title("Mean NTR per cell")

    sc2 = plot_umap_scalar(axes[1], X_umap, total_counts_vals, vmin=tot_lo, vmax=tot_hi)
    axes[1].set_title("Total counts per cell")

    sc3 = plot_umap_scalar(axes[2], X_umap, pct_mt_vals, vmin=mt_lo, vmax=mt_hi)
    axes[2].set_title("% mitochondrial counts")

    for ax in axes:
        ax.set_xlim(xlim_glob)
        ax.set_ylim(ylim_glob)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0)

    for sca, ax in zip([sc1, sc2, sc3], axes):
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(length=2)

    out.savefig_paper("umap_scalars_meanNTR_total_pctMT.svg")
    out.savefig_svg_for_word("umap_scalars_meanNTR_total_pctMT_for_word.svg")
    plt.show()


def plot_marker_umaps(adata_umap, X_umap, xlim_glob, ylim_glob, markers, out):
    markers = [g for g in markers if g in adata_umap.raw.var_names]
    if not markers:
        print("No marker genes present in adata_umap.raw.var_names. Skipping marker UMAPs.")
        return

    Xraw = adata_umap.raw[:, markers].X
    if sparse.issparse(Xraw):
        Xraw = Xraw.toarray()

    vmin = np.nanpercentile(Xraw, 1)
    vmax = np.nanpercentile(Xraw, 99)

    n = len(markers)
    cols = 3 if n >= 4 else n
    rows = int(np.ceil(n / 3)) if n >= 4 else 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.0 * cols + 0.6, 2.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    k = 0
    last_sca = None
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < n:
                vals = np.asarray(Xraw[:, k]).ravel()
                last_sca = ax.scatter(
                    X_umap[:, 0],
                    X_umap[:, 1],
                    c=vals,
                    s=6,
                    alpha=0.9,
                    cmap=PAPER_CMAP,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0,
                )
                ax.set_title(markers[k], fontsize=9)
                k += 1
            else:
                ax.axis("off")

            ax.set(xticks=[], yticks=[])
            ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
            ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
            ax.set_xlim(xlim_glob)
            ax.set_ylim(ylim_glob)
            ax.set_aspect("equal")
            ax.margins(0)

    if last_sca is not None:
        cbar = fig.colorbar(last_sca, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.ax.tick_params(length=2, labelsize=UMAP_LABEL_FS)

    out.savefig_paper("umap_markers_sharedscale.svg")
    out.savefig_svg_for_word("umap_markers_sharedscale_for_word.svg")
    plt.show()


def plot_transition_index_umap(adata_umap, X_umap, xlim_glob, ylim_glob, out):
    ti_vals = adata_umap.obs["transition_index"].to_numpy()

    with mpl.rc_context(
        {
            "font.size": 7.8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        sca = plot_umap_transition(ax, X_umap, ti_vals, s=6, alpha=0.9)
        ax.set_xlim(xlim_glob)
        ax.set_ylim(ylim_glob)
        ax.margins(0)
        ax.grid(False)

        cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.01)
        cb.ax.tick_params(length=2, labelsize=UMAP_LABEL_FS)

        counts = adata_umap.obs["cell_state"].value_counts().reindex(STATE_CATEGORIES).fillna(0).astype(int)
        props = (counts / counts.sum() * 100).round(1)
        txt = "\n".join([f"{k} ({props[k]:.1f}%)" for k in STATE_CATEGORIES])

        ax.text(
            0.5,
            -0.18,
            txt,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="0.35",
                linewidth=0.8,
                alpha=1.0,
            ),
        )
        fig.subplots_adjust(bottom=0.25)

    out.savefig_paper("umap_transition_index.svg")
    out.savefig_svg_for_word("umap_transition_index_for_word.svg")
    plt.show()


def plot_cell_state_umap(adata_umap, X_umap, xlim_glob, ylim_glob, out):
    state_labels = adata_umap.obs["cell_state"].astype(str).to_numpy()
    counts = adata_umap.obs["cell_state"].value_counts().reindex(STATE_CATEGORIES).fillna(0).astype(int)
    props = (counts / counts.sum() * 100).round(1)
    labels = {k: f"{k} ({props.loc[k]:.1f}%)" for k in STATE_CATEGORIES}

    plt.figure(figsize=(2.2, 2.2))
    ax = plt.gca()

    for lab in STATE_CATEGORIES:
        m = state_labels == lab
        ax.scatter(
            X_umap[m, 0],
            X_umap[m, 1],
            s=6,
            alpha=0.95,
            linewidths=0,
            c=SET1_COLORS[lab],
        )

    ax.set(xticks=[], yticks=[])
    ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
    ax.set_xlim(xlim_glob)
    ax.set_ylim(ylim_glob)
    ax.set_aspect("equal")
    ax.margins(0)
    ax.grid(False)

    txt = "\n".join([labels[k] for k in STATE_CATEGORIES])
    fig = plt.gcf()
    ax.text(
        0.5,
        -0.18,
        txt,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.8,
            alpha=1.0,
        ),
    )
    fig.subplots_adjust(bottom=0.25)

    out.savefig_paper("umap_cell_states.svg")
    out.savefig_svg_for_word("umap_cell_states_for_word.svg")
    plt.show()


def plot_half_life_vs_slam(merged_df, out, hl_max=24):
    x = merged_df["SLAM_half_life_hr"].to_numpy()
    y = merged_df["half_life_hr"].to_numpy()
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)

    plt.figure(figsize=(5.6, 4.8))
    sca = plt.scatter(x, y, c=z, cmap="viridis", s=12, alpha=0.85, edgecolors="none")
    plt.plot([0, hl_max], [0, hl_max], ls="--", c="gray")
    add_corr_box(plt.gca(), x, y, method="pearson", loc="bl", show_n=False)
    plt.xlabel("SLAM-seq Half-life (hr)")
    plt.ylabel("Estimated Half-life (hr)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, hl_max)
    plt.ylim(0, hl_max)
    cb = plt.colorbar(sca)
    cb.set_label("Local Density")
    plt.tight_layout()
    out.savefig_named("half_life_vs_SLAM_0-24h.svg", format="svg")
    plt.show()


def _boxplot_from_dict(ax, res, title, box_width=0.6):
    ks = sorted(res.keys())
    data = [res[k] if len(res[k]) else [np.nan] for k in ks]

    ax.boxplot(
        data,
        positions=ks,
        widths=[box_width] * len(ks),
        showfliers=True,
        whis=1.5,
        patch_artist=True,
        boxprops=dict(facecolor="white", linewidth=1),
        medianprops=dict(color="C1", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor="none",
            markeredgecolor="black",
            alpha=0.6,
        ),
    )
    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 1, max(ks) + 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cell number sampled")
    ax.set_ylabel("Pearson correlation")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    for x in ks:
        ax.axvline(x, color="0.92", lw=0.8, zorder=0)


def plot_stability_results(res_pluri, res_2c, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

    _boxplot_from_dict(axes[0], res_pluri, "Half-life stability — Pluripotent", box_width=0.6)
    _boxplot_from_dict(axes[1], res_2c, "Half-life stability — 2-cell-like", box_width=0.6)

    for ax, res in zip(axes, [res_pluri, res_2c]):
        ks = sorted(res.keys())
        med = [np.median(res[k]) if len(res[k]) else np.nan for k in ks]
        ax.plot(ks, med, marker="o", ms=3, lw=1, color="C1")

    plt.tight_layout()
    out.savefig_named("stability_half_life_pluri_vs_2Clike.svg", format="svg")
    plt.show()


def plot_rate_comparison(
    merged_df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    out_filename,
    out,
):
    plt.figure(figsize=(5.0, 5.0))

    sns.kdeplot(
        x=merged_df[x_col],
        y=merged_df[y_col],
        fill=True,
        cmap="Blues",
        bw_adjust=0.8,
        levels=50,
        thresh=0,
        alpha=0.55,
    )
    plt.scatter(
        merged_df[x_col],
        merged_df[y_col],
        s=8,
        alpha=0.4,
        color="black",
        edgecolors="none",
    )

    corr, pval = pearsonr(merged_df[x_col], merged_df[y_col])
    print(f"{title} Pearson correlation: {corr:.3f} (p = {pval:.2e})")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    add_corr_box(
        plt.gca(),
        merged_df[x_col].to_numpy(),
        merged_df[y_col].to_numpy(),
        loc="br",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    out.savefig_named(out_filename, format="svg")
    plt.show()


def plot_dropout_vs_ntr_rank(df_state_diag, states, out):
    fig, axes = plt.subplots(
        1,
        len(states),
        figsize=(len(states) * 5.6, 4.4),
        constrained_layout=True,
        sharey=True,
    )
    if len(states) == 1:
        axes = [axes]

    cmap = mpl.colormaps.get_cmap("viridis")
    norm = _Normalize(vmin=0, vmax=1)
    rng = np.random.default_rng(42)

    for ax, state in zip(axes, states):
        sub = df_state_diag[df_state_diag["state"] == state].copy()
        x = sub["ntr_rank_order"].to_numpy(dtype=float)
        y = sub["log2FC_centered"].to_numpy(dtype=float)

        x = x + rng.uniform(-0.35, 0.35, size=x.shape)

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        z = (z - z.min()) / (z.max() - z.min() + 1e-12)

        ax.scatter(x, y, c=z, cmap=cmap, s=12, alpha=0.85, edgecolors="none")
        sns.regplot(x=x, y=y, scatter=False, lowess=True, ax=ax, line_kws={"lw": 2})

        rho = sub[["ntr_rank_order", "log2FC_centered"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{state}\nSpearman ρ = {rho:.2f}")
        ax.axhline(0, ls="--", lw=1, c="gray")
        ax.set_xlabel("Gene Rank by NTR (High → Low)")

    axes[0].set_ylabel("Mean-centered log₂ FC (4sU / no4sU)")
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=np.ravel(axes).tolist(),
        fraction=0.02,
        pad=0.02,
    )
    cbar.set_label("Local Density")
    out.savefig_named("dropout_vs_NTRrank_by_state.svg", format="svg")
    plt.show()