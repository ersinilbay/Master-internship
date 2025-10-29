import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import pearsonr, gaussian_kde, spearmanr
from scipy import sparse
import scipy.sparse as sp
from matplotlib.colors import Normalize as _Normalize, TwoSlopeNorm as _TwoSlopeNorm
 
# Colormaps
PAPER_CMAP = mpl.cm.viridis
TRANSITION_CMAP = mpl.cm.plasma

# Paper export + style helpers
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"]  = 42


# === little correlation badge ===
def add_corr_box(ax, x, y, method="pearson", loc="br", show_n=False):
    """Draws a small rounded box with Pearson/Spearman on `ax`.
       loc ∈ {"tl","tr","bl","br"} = top/bottom-left/right."""
    import numpy as _np
    from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr

    m = _np.isfinite(x) & _np.isfinite(y)
    if m.sum() == 0:
        return
    if method == "spearman":
        r, _ = _spearmanr(x[m], y[m])
        text = f"Spearman \u03C1 = {r:.2f}"
    else:
        r, _ = _pearsonr(x[m], y[m])
        text = f"Pearson r = {r:.2f}"
    if show_n:
        text += f"\n n = {int(m.sum())}"

    pos = {
        "tl": (0.02, 0.98, dict(ha="left",  va="top")),
        "tr": (0.98, 0.98, dict(ha="right", va="top")),
        "bl": (0.02, 0.02, dict(ha="left",  va="bottom")),
        "br": (0.95, 0.03, dict(ha="right", va="bottom")),
    }[loc]

    ax.text(
        pos[0], pos[1], text, transform=ax.transAxes,
        bbox=dict(boxstyle="square,pad=0.18",
                  facecolor="white", edgecolor="0.35",
                  linewidth=0.8, alpha=0.9),
        **pos[2]
    )

# ---- global figure style ----
PAPER_FIG_DPI = 300
mpl.rcParams.update({
    "font.size": 7.8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": PAPER_FIG_DPI,
})
# Fixed Set1 colors for state bullets (used by legends)
set1_colors = {
    "Pluripotent":  "#E41A1C",
    "Intermediate": "#377EB8",
    "2-cell like":  "#4DAF4A",
}

UMAP_LABEL_FS = 7                      # mid-size for both marker & transition UMAP axes



def savefig_paper(filename_svg: str):
    base, ext = os.path.splitext(filename_svg)
    name = f"{base}_paper.svg" if ext.lower() != ".svg" else f"{base}_paper{ext}"
    savefig_named(name, format="svg")

def savefig_svg_for_word(filename_svg: str):
    old = mpl.rcParams['svg.fonttype']
    mpl.rcParams['svg.fonttype'] = 'path'  # outline text for Word
    savefig_named(filename_svg, format="svg")
    mpl.rcParams['svg.fonttype'] = old

def plot_umap_scalar(ax, X_umap, vals, vmin=None, vmax=None, s=6, alpha=0.9):
    norm = _Normalize(vmin=vmin, vmax=vmax)
    sca = ax.scatter(X_umap[:,0], X_umap[:,1], c=vals, s=s, alpha=alpha,
                     cmap=PAPER_CMAP, norm=norm, linewidths=0)
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
    ax.set_aspect("equal", adjustable="box")
    return sca
# ======================= OUTPUT HELPERS (folder + suffix) =======================
OUT_DIR = "_rep1_fix"
SUFFIX = "_rep1_fix"
os.makedirs(OUT_DIR, exist_ok=True)

def add_suffix(filename: str, suffix: str = SUFFIX) -> str:
    base, ext = os.path.splitext(filename)
    return f"{base}{suffix}{ext}"

def out_path(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)

def savefig_named(filename: str, **kwargs):
    """Save the current matplotlib figure to OUT_DIR with suffix."""
    plt.savefig(out_path(add_suffix(filename)), bbox_inches="tight", **kwargs)

def to_csv_named(df: pd.DataFrame, filename: str, **kwargs):
    df.to_csv(out_path(add_suffix(filename)), **kwargs)

def write_h5ad_named(adata: sc.AnnData, filename: str, **kwargs):
    adata.write(out_path(add_suffix(filename)), **kwargs)

print(f"All outputs will be written to: {OUT_DIR} with suffix {SUFFIX}")

# ----------------------- Plot style (consistent layout) -----------------------
FIGSIZE = (7.5, 5.5)
INTERACTIVE_DPI = 110

plt.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": INTERACTIVE_DPI,   # <-- smaller windows
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})
sns.set_context("notebook", font_scale=1.0)

def plot_umap_transition(ax, X_umap, vals, s=6, alpha=0.9):
    lo, hi = np.nanpercentile(vals, [1, 99])
    m = float(max(abs(lo), abs(hi)))
    norm = _TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=+m)
    sca = ax.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=vals, s=s, alpha=alpha,
        cmap=TRANSITION_CMAP, norm=norm, linewidths=0
    )
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
    ax.set_aspect("equal", adjustable="box")
    return sca

def add_cut_lines_minimal():
    """Just red dashed cut lines + small red labels. No shading, no boxes."""
    axes = plt.gcf().axes                     # [n_genes, total_counts, pct_counts_mt]
    CUT  = "#B22222"                          # firebrick
    DASH = (0, (7, 4))                        # long dash pattern
    cuts = [
        [("≥ 200", 200), ("≤ 2500", 2500)],   # panel 1
        [("≤ 5000", 5000)],                   # panel 2
        [("≤ 5", 5)],                         # panel 3
    ]
    for ax, arr in zip(axes, cuts):
        for txt, y in arr:
            ax.axhline(y, color=CUT, lw=2.4, ls=DASH, zorder=10)
            ax.text(0.01, y, txt,
                    transform=ax.get_yaxis_transform(),
                    color=CUT, fontsize=9, ha="left", va="bottom", zorder=11)

# ============================= Step 1: Load data ================================
c_file = "/Users/ersinilbay/Downloads/mESC-WT-rep1_C.txt"
t_file = "/Users/ersinilbay/Downloads/mESC-WT-rep1_T.txt"

c_df = pd.read_csv(c_file, sep="\t", index_col=0)
t_df = pd.read_csv(t_file, sep="\t", index_col=0)

C = c_df.values.T
T = t_df.values.T
total = C + T

adata_initial = sc.AnnData(
    X=total.copy(),
    var=pd.DataFrame(index=c_df.index),
    obs=pd.DataFrame(index=c_df.columns)
)

# Add layers
adata_initial.layers["C"] = C
adata_initial.layers["T"] = T
adata_initial.layers["total"] = total

# Add NTR and QC metrics
ntr = np.where(total != 0, C / total, 0)
adata_initial.layers["ntr"] = ntr
adata_initial.obs["mean_ntr"] = ntr.mean(axis=1)
adata_initial.var["mt"] = adata_initial.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata_initial, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

adata_pre = adata_initial.copy()  # snapshot BEFORE any cell/gene filtering

# =================== Step 2: Quality control filtering  ===================
sc.pp.filter_cells(adata_initial, min_genes=200)
sc.pp.filter_genes(adata_initial, min_cells=3)
adata_initial = adata_initial[adata_initial.obs['total_counts'] < 5000, :]
adata_initial = adata_initial[adata_initial.obs['n_genes_by_counts'] < 2500, :]
adata_initial = adata_initial[adata_initial.obs['pct_counts_mt'] < 5, :]

# Recompute QC metrics & NTR after filtering (for plots/consistency)
adata_initial.var["mt"] = adata_initial.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata_initial, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
ntr = np.where(adata_initial.layers["total"] != 0,
               adata_initial.layers["C"] / adata_initial.layers["total"], 0)
adata_initial.layers["ntr"] = ntr
adata_initial.obs["mean_ntr"] = ntr.mean(axis=1)

# ---------- QC violins (POST-QC) with cutoff overlays ----------
sc.pl.violin(
    adata_initial,
    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    jitter=0.4, log=False, multi_panel=True, show=False
)
# Relabel each panel
for ax, lab in zip(plt.gcf().axes,
                   ["Genes detected (UMIs)", "Total UMIs", "Mitochondrial UMIs (%)"]):
    ax.set_ylabel(lab)

add_cut_lines_minimal()

plt.tight_layout()
savefig_named("violin_plots_POSTQC_cutoffs.svg", format="svg")
plt.show()

# ---------- QC violins (PRE-QC snapshot) with the same cutoffs ----------
# (adata_pre already exists above and has QC fields)
sc.pl.violin(
    adata_pre,
    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    jitter=0.4, log=False, multi_panel=True, show=False
)
# Relabel each panel
for ax, lab in zip(plt.gcf().axes,
                   ["Genes detected (UMIs)", "Total UMIs", "Mitochondrial UMIs (%)"]):
    ax.set_ylabel(lab)

add_cut_lines_minimal()
plt.tight_layout()
savefig_named("violin_plots_PREQC_cutoffs.svg", format="svg")
plt.show()

# QC scatter
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
sc.pl.scatter(adata_initial, x='total_counts', y='n_genes_by_counts', ax=axes[0],
              title="Genes vs total UMIs", size=20, show=False)
sc.pl.scatter(adata_initial, x='total_counts', y='pct_counts_mt', ax=axes[1],
              title="% MT vs total UMIs", size=20, show=False)

axes[0].set_xlabel("Total UMIs")
axes[0].set_ylabel("Genes detected (UMIs)")
axes[1].set_xlabel("Total UMIs")
axes[1].set_ylabel("Mitochondrial UMIs (%)")

# --- make each panel square (same visual box height & width)
for ax in axes:
    ax.set_box_aspect(1.0)          # if mpl<3.3, use: ax.set_aspect('equal', adjustable='box')

# --- add Pearson r badge in bottom-right
add_corr_box(
    axes[0],
    adata_initial.obs['total_counts'].values,
    adata_initial.obs['n_genes_by_counts'].values,
    loc='br'
)
add_corr_box(
    axes[1],
    adata_initial.obs['total_counts'].values,
    adata_initial.obs['pct_counts_mt'].values,
    loc='br'
)

plt.tight_layout()
savefig_named("scatter_plots.svg", format="svg")
plt.show()

# ======================= Step 3: Split working objects ==========================
adata_qc_raw = adata_initial.copy()    # For half-life, synthesis, degradation (unnormalized)
adata_umap = adata_initial.copy()      # For normalization/UMAP/clustering/plots

# ======================= FULL (all genes x all cells) EXPORTS =======================
# Uses adata_qc_raw so it's QC-filtered cells but NO HVG filtering.

# Extract C (new RNA) and T (old RNA) layers and convert to dense if needed
C_full = adata_qc_raw.layers["C"]
T_full = adata_qc_raw.layers["T"]
C_full = C_full.toarray() if sparse.issparse(C_full) else C_full
T_full = T_full.toarray() if sparse.issparse(T_full) else T_full

# Gene x cell CSVs
newrna_full_df = pd.DataFrame(C_full.T, index=adata_qc_raw.var_names, columns=adata_qc_raw.obs_names)
oldrna_full_df = pd.DataFrame(T_full.T, index=adata_qc_raw.var_names, columns=adata_qc_raw.obs_names)

to_csv_named(newrna_full_df, "newrna_full.csv")
to_csv_named(oldrna_full_df, "oldrna_full.csv")

print("Exported FULL matrices: newrna_full.csv and oldrna_full.csv")

# ================= UMAP + Leiden + manual state annotation =====================
# Normalization + HVGs (Seurat v3 style)
sc.pp.normalize_total(adata_umap, target_sum=1e4)
sc.pp.log1p(adata_umap)

# Store full gene expression before HVG filtering
adata_umap.raw = adata_umap.copy()

sc.pp.highly_variable_genes(adata_umap, flavor="seurat_v3", n_top_genes=2000)
adata_umap = adata_umap[:, adata_umap.var.highly_variable]

# Scaling + PCA
sc.pp.scale(adata_umap, max_value=10)
sc.tl.pca(adata_umap, n_comps=50)
sc.pl.pca_variance_ratio(adata_umap, log=True, show=False)
plt.gca().set_ylabel("Explained variance ratio (log10)")
plt.tight_layout()
savefig_named("pcas.svg", format="svg")
plt.show()

# Neighbors + UMAP + Leiden
sc.pp.neighbors(adata_umap, n_neighbors=30, n_pcs=30)
try:
    sc.tl.umap(adata_umap, min_dist=0.05, spread=1.2, random_state=42)
except TypeError:
    sc.tl.umap(adata_umap)

X_umap = adata_umap.obsm["X_umap"]
x = X_umap[:,0]; y = X_umap[:,1]
x_lo, x_hi = np.percentile(x, [0.2, 99.8])
y_lo, y_hi = np.percentile(y, [0.2, 99.8])
cx, cy = (x_lo+x_hi)/2, (y_lo+y_hi)/2
side = max(x_hi-x_lo, y_hi-y_lo) * 1.18
xlim_glob = (cx - side/2, cx + side/2)
ylim_glob = (cy - side/2, cy + side/2)

sc.tl.leiden(adata_umap, resolution=2)

# ---------- QC scalar UMAPs ----------
mean_ntr_vals     = adata_umap.obs["mean_ntr"].to_numpy()
total_counts_vals = adata_umap.obs["total_counts"].to_numpy()
pct_mt_vals       = adata_umap.obs["pct_counts_mt"].to_numpy()

# robust 1–99% limits
ntr_lo, ntr_hi = np.nanpercentile(mean_ntr_vals, [1, 99])
tot_lo, tot_hi = np.nanpercentile(total_counts_vals, [1, 99])
mt_lo,  mt_hi  = np.nanpercentile(pct_mt_vals,   [1, 99])

fig, axes = plt.subplots(
    1, 3, figsize=(6.6, 2.2),
    constrained_layout=True, sharex=True, sharey=True
)

sc1 = plot_umap_scalar(axes[0], X_umap, mean_ntr_vals,  vmin=ntr_lo, vmax=ntr_hi); axes[0].set_title("Mean NTR per cell")
sc2 = plot_umap_scalar(axes[1], X_umap, total_counts_vals, vmin=tot_lo, vmax=tot_hi); axes[1].set_title("Total counts per cell")
sc3 = plot_umap_scalar(axes[2], X_umap, pct_mt_vals,      vmin=mt_lo,  vmax=mt_hi);  axes[2].set_title("% mitochondrial counts")

for ax in axes:
    ax.set_xlim(xlim_glob); ax.set_ylim(ylim_glob)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

# slim per-panel colorbars
for sca, ax in zip([sc1, sc2, sc3], axes):
    cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(length=2)

savefig_paper("umap_scalars_meanNTR_total_pctMT.svg")
savefig_svg_for_word("umap_scalars_meanNTR_total_pctMT_for_word.svg")
plt.show()

markers = ["Nanog","Zfp42","Myc","Zscan4c","Sp110"]
markers = [g for g in markers if g in adata_umap.raw.var_names]  # keep present ones

# get raw log1p-normalized values as dense
Xraw = adata_umap.raw[:, markers].X
if sparse.issparse(Xraw): Xraw = Xraw.toarray()

# single shared color scale (1st–99th pct across ALL markers)
vmin = np.nanpercentile(Xraw, 1)
vmax = np.nanpercentile(Xraw, 99)

n = len(markers)

# Force 3 across on the first row, spill the rest onto the second row
cols = 3 if n >= 4 else n
rows = int(np.ceil(n / 3)) if n >= 4 else 1

fig, axes = plt.subplots(
    rows, cols,
    figsize=(2.0*cols + 0.6, 2.0*rows),
    squeeze=False, constrained_layout=True
)

k = 0
for r in range(rows):
    for c in range(cols):
        ax = axes[r, c]
        if k < n:
            vals = np.asarray(Xraw[:, k]).ravel()
            sca = ax.scatter(X_umap[:,0], X_umap[:,1], c=vals, s=6, alpha=0.9,
                             cmap=PAPER_CMAP, vmin=vmin, vmax=vmax, linewidths=0)
            ax.set_title(markers[k], fontsize=9)
            k += 1
        else:
            ax.axis("off")
        ax.set(xticks=[], yticks=[])
        ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
        ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
        ax.set_xlim(xlim_glob); ax.set_ylim(ylim_glob); ax.set_aspect("equal"); ax.margins(0)

# one shared colorbar
cbar = fig.colorbar(sca, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
cbar.ax.tick_params(length=2, labelsize=UMAP_LABEL_FS)

savefig_paper("umap_markers_sharedscale.svg")
savefig_svg_for_word("umap_markers_sharedscale_for_word.svg")
plt.show()

# --------------------------- Qiu-based annotation ------------------------------
twoC_markers = ["Zscan4c", "Sp110", "Dppa2", "Gm4340", "Gm4981"]
pluri_markers = ["Nanog", "Zfp42", "Myc", "Esrrb"]

available_2C = [g for g in twoC_markers if g in adata_umap.raw.var_names]
available_pluri = [g for g in pluri_markers if g in adata_umap.raw.var_names]
print(f"Using 2C markers: {available_2C}")
print(f"Using pluripotent markers: {available_pluri}")

_2c_vals = np.asarray(adata_umap.raw[:, available_2C].X.mean(axis=1)).ravel() if len(available_2C) > 0 else np.zeros(adata_umap.n_obs)
_pluri_vals = np.asarray(adata_umap.raw[:, available_pluri].X.mean(axis=1)).ravel() if len(available_pluri) > 0 else np.zeros(adata_umap.n_obs)

adata_umap.obs["2C_score"] = _2c_vals
adata_umap.obs["Pluri_score"] = _pluri_vals
adata_umap.obs["transition_index"] = adata_umap.obs["2C_score"] - adata_umap.obs["Pluri_score"]

# Target proportions from Qiu et al. [Pluripotent, Intermediate, 2-cell like]
target = np.array([98.3, 1.0, 0.7])
best_score = np.inf
best_cuts = (0.0, 0.0)

for inter_cut in np.linspace(-1, 1, 200):
    for c2c_cut in np.linspace(inter_cut + 0.1, 2, 200):
        cats = np.full(len(adata_umap), "Pluripotent", dtype=object)
        ti = adata_umap.obs["transition_index"].values
        cats[(ti > inter_cut) & (ti <= c2c_cut)] = "Intermediate"
        cats[ti > c2c_cut] = "2-cell like"

        vals, counts = np.unique(cats, return_counts=True)
        props = np.zeros(3)
        for i, label in enumerate(["Pluripotent", "Intermediate", "2-cell like"]):
            if label in vals:
                props[i] = 100 * counts[vals.tolist().index(label)] / len(ti)
        mse = np.mean((props - target) ** 2)
        if mse < best_score:
            best_score = mse
            best_cuts = (inter_cut, c2c_cut)

print(f"\nBest match with article at:\ninter_cutoff = {best_cuts[0]:.3f}, 2C_cutoff = {best_cuts[1]:.3f}")

# Adopt adjusted cutoffs
INTER_CUTOFF = 0.005 #to match Qiu
C2C_CUTOFF = 0.819 #to match Qiu

adata_umap.obs["cell_state"] = "Pluripotent"
adata_umap.obs.loc[adata_umap.obs["transition_index"] > C2C_CUTOFF, "cell_state"] = "2-cell like"
adata_umap.obs.loc[
    (adata_umap.obs["transition_index"] > INTER_CUTOFF) &
    (adata_umap.obs["transition_index"] <= C2C_CUTOFF),
"cell_state"
] = "Intermediate"

adata_umap.obs["cell_state"] = pd.Categorical(
    adata_umap.obs["cell_state"],
    categories=["Pluripotent", "Intermediate", "2-cell like"],
    ordered=True
)

# ---------- Transition index UMAP (journal layout, diverging, centered at 0) ----------
ti_vals = adata_umap.obs["transition_index"].to_numpy()

with mpl.rc_context({
    "font.size": 7.8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6
}):
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    sca = plot_umap_transition(ax, X_umap, ti_vals, s=6, alpha=0.9)
    ax.set_xlim(xlim_glob);
    ax.set_ylim(ylim_glob);
    ax.margins(0);
    ax.grid(False)

    cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.01)

    # --- percentages summary by panel ---
    cats = ["Pluripotent", "Intermediate", "2-cell like"]
    counts = adata_umap.obs["cell_state"].value_counts().reindex(cats).fillna(0).astype(int)
    props = (counts / counts.sum() * 100).round(1)

    lines = [f"{k} ({props[k]:.1f}%)" for k in cats]
    txt = "\n".join(lines)

    # place a white, square-corner box centered *below* the axes
    ax.text(
        0.5, -0.18, txt,
        transform=ax.transAxes, ha="center", va="top", fontsize=8,
        bbox=dict(boxstyle="square,pad=0.35",
                  facecolor="white", edgecolor="0.35",
                  linewidth=0.8, alpha=1.0)
    )
    # leave room for the box
    fig.subplots_adjust(bottom=0.25)

    cb.ax.tick_params(length=2, labelsize=UMAP_LABEL_FS)


savefig_paper("umap_transition_index.svg")
savefig_svg_for_word("umap_transition_index_for_word.svg")
plt.show()



# ---------- Cell-state UMAP (Set1 palette, legend shows counts; single panel) ----------
state_labels = adata_umap.obs["cell_state"].astype(str).to_numpy()
cats  = ["Pluripotent","Intermediate","2-cell like"]
counts = adata_umap.obs["cell_state"].value_counts().reindex(cats).fillna(0).astype(int)
props  = (counts / counts.sum() * 100).round(1)
labels = {k: f"{k} ({props.loc[k]:.1f}%)" for k in cats}



plt.figure(figsize=(2.2, 2.2))
ax = plt.gca()
for lab in cats:
    m = (state_labels == lab)
    ax.scatter(X_umap[m,0], X_umap[m,1], s=6, alpha=0.95, linewidths=0,
               c=set1_colors[lab])
ax.set(xticks=[], yticks=[])
ax.set_xlabel("UMAP1", fontsize=UMAP_LABEL_FS)
ax.set_ylabel("UMAP2", fontsize=UMAP_LABEL_FS)
ax.set_xlim(xlim_glob); ax.set_ylim(ylim_glob); ax.set_aspect("equal"); ax.margins(0); ax.grid(False)


# text-only percentages box under the plot
txt = "\n".join([labels[k] for k in cats])
fig = plt.gcf()
ax.text(
    0.5, -0.18, txt,
    transform=ax.transAxes, ha="center", va="top", fontsize=8,
    bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="0.35", linewidth=0.8, alpha=1.0)
)
fig.subplots_adjust(bottom=0.25)

savefig_paper("umap_cell_states.svg")
savefig_svg_for_word("umap_cell_states_for_word.svg")
plt.show()



# Save UMAP + states (has obsm["X_umap"], obs["cell_state"], transition_index, etc.)
write_h5ad_named(adata_umap, "adata_umap_with_states.h5ad", compression="gzip")



# Split AnnData by state (renamed)
pluripotent_adata = adata_umap[adata_umap.obs["cell_state"] == "Pluripotent"].copy()
intermediate_adata = adata_umap[adata_umap.obs["cell_state"] == "Intermediate"].copy()
twocelllike_adata = adata_umap[adata_umap.obs["cell_state"] == "2-cell like"].copy()

# Check percentages
print((adata_umap.obs["cell_state"].value_counts(normalize=True) * 100).round(2))
counts = adata_umap.obs["cell_state"].value_counts()
percentages = (counts / counts.sum()) * 100
print(percentages)

# =================== Step 5: Gene-level kinetic calculations ===================
T_label = 4  # hours

new_raw   = np.asarray(adata_qc_raw.layers["C"].sum(axis=0), dtype=float).flatten()
raw_total = np.asarray((adata_qc_raw.layers["C"] + adata_qc_raw.layers["T"]).sum(axis=0), dtype=float).flatten()

# Stable ratio + masking (avoids divide-by-zero and log(0) warnings)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.divide(new_raw, raw_total, out=np.zeros_like(new_raw), where=raw_total > 0)
ratio = np.clip(ratio, 0.0, 1.0 - 1e-12)
valid = (raw_total > 0) & (ratio > 0) & (ratio < 1.0 - 1e-12)

# Half-life, degradation, synthesis (stable forms)
half_life = np.full_like(raw_total, np.nan, dtype=float)
half_life[valid] = -T_label * np.log(2) / np.log1p(-ratio[valid])  # use log1p for stability

deg_rate = np.full_like(half_life, np.nan, dtype=float)
deg_rate[valid] = np.log(2) / half_life[valid]

new_mean = np.asarray(adata_qc_raw.layers["C"].mean(axis=0), dtype=float).flatten()
exp_term = np.exp(-deg_rate * T_label)
synth_rate = np.full_like(deg_rate, np.nan, dtype=float)
ok = np.isfinite(deg_rate) & (exp_term != 1)
synth_rate[ok] = (new_mean[ok] * deg_rate[ok]) / (1 - exp_term[ok])

adata_qc_raw.var["half_life_hr"] = half_life
adata_qc_raw.var["deg_rate"]     = deg_rate
adata_qc_raw.var["synth_rate"]   = synth_rate

print(adata_qc_raw.var["half_life_hr"].head())

# Exports (into OUT_DIR with suffix)
kinetics = adata_qc_raw.var[["half_life_hr", "deg_rate", "synth_rate"]].copy()
kinetics.index.name = "Gene"
kinetics = kinetics.replace([np.inf, -np.inf], np.nan)

to_csv_named(kinetics, "gene_kinetics_all.csv")
kinetics_valid = kinetics.dropna().query("half_life_hr > 0 and half_life_hr < 24")
to_csv_named(kinetics_valid, "gene_kinetics_valid_0-24h.csv")

# === SLAM-seq half-life vs our estimated half-life (restrict to 0–24 h) ===
HL_MAX = 24

# Load SLAM (T→C) half-lives
slam_df = pd.read_excel(
    "/Users/ersinilbay/Downloads/41592_2017_BFnmeth4435_MOESM4_ESM.xls",
    engine="xlrd"
)
# fix commas-as-decimals and filter quality
slam_df["Half-life (h)"] = slam_df["Half-life (h)"].astype(str).str.replace(",", ".").astype(float)
slam_df["Rsquare"]       = slam_df["Rsquare"].astype(str).str.replace(",", ".").astype(float)
slam = slam_df.loc[slam_df["Rsquare"] > 0.4, ["Name", "Half-life (h)"]].rename(
    columns={"Name": "Gene", "Half-life (h)": "SLAM_half_life_hr"}
)
slam = slam.drop_duplicates(subset="Gene")

# Our per-gene half-lives from adata_qc_raw
our = adata_qc_raw.var[["half_life_hr"]].reset_index().rename(columns={"index": "Gene"})

m = pd.merge(slam, our, on="Gene", how="inner")
m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["SLAM_half_life_hr", "half_life_hr"])
# keep sane range only
m = m[(m["SLAM_half_life_hr"] > 0) & (m["SLAM_half_life_hr"] <= HL_MAX) &
      (m["half_life_hr"] > 0)      & (m["half_life_hr"]      <= HL_MAX)]

print(f"[half-life corr] n={len(m)} genes after filtering to 0–{HL_MAX} h")

# Scatter with density + y=x
x = m["SLAM_half_life_hr"].to_numpy()
y = m["half_life_hr"].to_numpy()
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
z = (z - z.min()) / (z.max() - z.min() + 1e-12)

plt.figure(figsize=(5.6, 4.8))
sc = plt.scatter(x, y, c=z, cmap="viridis", s=12, alpha=0.85, edgecolors="none")
plt.plot([0, HL_MAX], [0, HL_MAX], ls="--", c="gray")
r, p = pearsonr(x, y)
add_corr_box(plt.gca(), x, y, method="pearson", loc="bl", show_n=False)
plt.xlabel("SLAM-seq Half-life (hr)")
plt.ylabel("Estimated Half-life (hr)")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(0, HL_MAX); plt.ylim(0, HL_MAX)
cb = plt.colorbar(sc); cb.set_label("Local Density")
plt.tight_layout()
savefig_named("half_life_vs_SLAM_0-24h.svg", format="svg")
plt.show()

# -------- Helpers for subsampling stability (hold-out reference) --------
def _half_life_from_mask(mask_bool):
    """Pseudo-bulk half-life per gene from a boolean cell mask."""
    Ch = adata_qc_raw.layers["C"]; Th = adata_qc_raw.layers["T"]
    if sparse.issparse(Ch): Ch = Ch.toarray()
    if sparse.issparse(Th): Th = Th.toarray()
    Csum = Ch[mask_bool, :].sum(axis=0).astype(float)
    Tsum = Th[mask_bool, :].sum(axis=0).astype(float)
    Tot  = Csum + Tsum
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(Csum, Tot, out=np.zeros_like(Csum), where=Tot > 0)
    ratio = np.clip(ratio, 0.0, 1.0 - 1e-12)
    hl = np.full_like(Tot, np.nan, dtype=float)
    ok = (Tot > 0) & (ratio > 0) & (ratio < 1.0 - 1e-12)
    hl[ok] = -T_label * np.log(2) / np.log1p(-ratio[ok])
    return hl

def _subsample_stability_holdout(state, k_grid, B=50, seed=42):
    """For each k: draw k cells, compute half-lives; compare to reference built from OTHER state cells."""
    rng = np.random.default_rng(seed)
    # make sure states are on the raw object and aligned
    adata_qc_raw.obs["cell_state"] = adata_umap.obs["cell_state"].reindex(adata_qc_raw.obs_names)
    idx_state = np.where(adata_qc_raw.obs["cell_state"].values == state)[0]
    n = idx_state.size
    if n < 3:
        print(f"[skip] {state}: too few cells ({n})."); return {}

    ks = [k for k in k_grid if k <= n - 2]  # leave ≥2 cells for the hold-out ref
    results = {k: [] for k in ks}

    for k in ks:
        for _ in range(B):
            take = rng.choice(idx_state, size=k, replace=False)
            ref_mask = np.zeros(adata_qc_raw.n_obs, dtype=bool); ref_mask[idx_state] = True; ref_mask[take] = False
            sub_mask = np.zeros_like(ref_mask); sub_mask[take] = True

            hl_ref = _half_life_from_mask(ref_mask)
            hl_sub = _half_life_from_mask(sub_mask)
            ok = np.isfinite(hl_ref) & np.isfinite(hl_sub)
            if ok.sum() >= 50:
                r, _ = pearsonr(hl_ref[ok], hl_sub[ok])
                results[k].append(r)
    return results
# -------- End helpers --------


# ===== CALL: subsampling stability (Pluripotent & 2-cell like) =====
res_pluri = _subsample_stability_holdout(
    "Pluripotent",
    k_grid=(5,10,20,30,40,60,80,120,200,400),
    B=30
)
res_2c = _subsample_stability_holdout(
    "2-cell like",
    k_grid=(3,5,8,10,12),   # function will drop k that violate hold-out
    B=200                   # more resampling for tiny n
)

# ---- Plot (boxplots over k) ----
def _boxplot_from_dict(ax, res, title, box_width=0.6):
    ks = sorted(res.keys())
    data = [res[k] if len(res[k]) else [np.nan] for k in ks]

    ax.boxplot(
        data,
        positions=ks,
        widths=[box_width] * len(ks),   # <-- constant, narrow boxes
        showfliers=True,
        whis=1.5,
        patch_artist=True,
        boxprops=dict(facecolor="white", linewidth=1),
        medianprops=dict(color="C1", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        flierprops=dict(marker='o', markersize=3, markerfacecolor='none',
                        markeredgecolor='black', alpha=0.6)
    )
    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 1, max(ks) + 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cell number sampled")
    ax.set_ylabel("Pearson correlation")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    # vertical guides
    for x in ks:
        ax.axvline(x, color="0.92", lw=0.8, zorder=0)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

# constant box width for both panels
_boxplot_from_dict(axes[0], res_pluri, "Half-life stability — Pluripotent", box_width=0.6)
_boxplot_from_dict(axes[1], res_2c,    "Half-life stability — 2-cell-like", box_width=0.6)

# median trend line
for ax, res in zip(axes, [res_pluri, res_2c]):
    ks = sorted(res.keys())
    med = [np.median(res[k]) if len(res[k]) else np.nan for k in ks]
    ax.plot(ks, med, marker='o', ms=3, lw=1, color='C1')

plt.tight_layout()
savefig_named("stability_half_life_pluri_vs_2Clike.svg", format="svg")
plt.show()

# ===== END stability plot =====

# =================== Deg & Synth rate comparisons (Qiu) ========================
qiu_df2 = pd.read_excel("/Users/ersinilbay/Downloads/scNTseq_params.xlsx",
                        sheet_name="Supplementary Table 5")
qiu_df2 = qiu_df2[["Gene", "Degradation_rate_Pluripotent", "Synthesis_rate_Pluripotent"]].dropna()
qiu_df2 = qiu_df2.replace([np.inf, -np.inf], np.nan).dropna()

my_deg = adata_qc_raw.var[["deg_rate"]].replace([np.inf, -np.inf], np.nan).dropna()
my_deg = my_deg.reset_index().rename(columns={"index": "Gene", "deg_rate": "Our_deg_rate"})
merged_deg = pd.merge(qiu_df2, my_deg, on="Gene").rename(
    columns={"Degradation_rate_Pluripotent": "scNTseq_deg_rate"}
)
merged_deg = merged_deg[(merged_deg["scNTseq_deg_rate"] <= 1)]

plt.figure(figsize=(5.0, 5.0))  # square canvas
# blue KDE cloud + points
sns.kdeplot(
    x=merged_deg["scNTseq_deg_rate"], y=merged_deg["Our_deg_rate"],
    fill=True, cmap="Blues", bw_adjust=0.8, levels=50, thresh=0, alpha=0.55
)
plt.scatter(merged_deg["scNTseq_deg_rate"], merged_deg["Our_deg_rate"],
            s=8, alpha=0.4, color="black", edgecolors="none")

corr, pval = pearsonr(merged_deg["scNTseq_deg_rate"], merged_deg["Our_deg_rate"])
print(f"Degradation rate Pearson correlation: {corr:.3f} (p = {pval:.2e})")
plt.xlabel("scNTseq degradation rate (hr⁻¹)")
plt.ylabel("Estimated degradation rate (hr⁻¹)")
plt.title("Degradation Rate Comparison")
add_corr_box(
    plt.gca(),
    merged_deg["scNTseq_deg_rate"].to_numpy(),
    merged_deg["Our_deg_rate"].to_numpy(),
    loc="br"
)

plt.xlim(0, 1); plt.ylim(0, 1); plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
savefig_named("degradation_rate_comparison.svg", format="svg")
plt.show()


my_synth = adata_qc_raw.var[["synth_rate"]].replace([np.inf, -np.inf], np.nan).dropna()
my_synth = my_synth.reset_index().rename(columns={"index": "Gene", "synth_rate": "Our_synth_rate"})
merged_synth = pd.merge(qiu_df2, my_synth, on="Gene").rename(
    columns={"Synthesis_rate_Pluripotent": "scNTseq_synth_rate"}
)

plt.figure(figsize=(5.0, 5.0))  # square canvas
# blue KDE cloud + points
sns.kdeplot(
    x=merged_synth["scNTseq_synth_rate"], y=merged_synth["Our_synth_rate"],
    fill=True, cmap="Blues", bw_adjust=0.8, levels=50, thresh=0, alpha=0.55
)
plt.scatter(merged_synth["scNTseq_synth_rate"], merged_synth["Our_synth_rate"],
            s=8, alpha=0.4, color="black", edgecolors="none")

corr, pval = pearsonr(merged_synth["scNTseq_synth_rate"], merged_synth["Our_synth_rate"])
print(f"Synthesis rate Pearson correlation: {corr:.3f} (p = {pval:.2e})")
plt.xlabel("scNTseq synthesis rate (hr⁻¹)")
plt.ylabel("Estimated synthesis rate (hr⁻¹)")
plt.title("Synthesis Rate Comparison")
add_corr_box(
    plt.gca(),
    merged_synth["scNTseq_synth_rate"].to_numpy(),
    merged_synth["Our_synth_rate"].to_numpy(),
    loc="br"
)
plt.xlim(0, 1); plt.ylim(0, 1); plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
savefig_named("synthesis_rate_comparison.svg", format="svg")
plt.show()


# ===================== Per-state 4sU dropout diagnostics =======================

# Keep gene symbols in var for easy joins later
adata_qc_raw.var["gene_symbol"] = adata_qc_raw.var_names

# Save raw counts + layers + kinetics + states (everything the next script needs)
write_h5ad_named(adata_qc_raw, "adata_qc_raw_with_kinetics_and_states.h5ad", compression="gzip")

# ================= Per-state AnnData exports (all genes) ==================

# Small slug for filenames (reuse your mapping)
slug = {"Pluripotent": "pluripotent", "Intermediate": "intermediate", "2-cell like": "2cell_like"}

# Optional: keep layers compact on disk
for L in ["C", "T", "total", "ntr"]:
    if L in adata_qc_raw.layers and adata_qc_raw.layers[L].dtype != np.float32:
        adata_qc_raw.layers[L] = adata_qc_raw.layers[L].astype(np.float32)

for state in adata_qc_raw.obs["cell_state"].cat.categories:
    mask = (adata_qc_raw.obs["cell_state"] == state).values
    adata_state = adata_qc_raw[mask, :].copy()

    if "total" not in adata_state.layers:
        adata_state.layers["total"] = adata_state.layers["C"] + adata_state.layers["T"]
    if "ntr" not in adata_state.layers:
        denom = np.clip(adata_state.layers["total"], 1e-12, None)
        adata_state.layers["ntr"] = adata_state.layers["C"] / denom

    tag = slug[state]
    write_h5ad_named(adata_state, f"adata_{tag}_full_allgenes.h5ad", compression="gzip")
    print(f"Wrote per-state AnnData (ALL genes): adata_{tag}_full_allgenes.h5ad")

    # CSVs per state (ALL genes) — inside the loop
    M_total = adata_state.layers["total"]
    M_ntr   = adata_state.layers["ntr"]
    if sp.issparse(M_total): M_total = M_total.toarray()
    if sp.issparse(M_ntr):   M_ntr   = M_ntr.toarray()

    total_df = pd.DataFrame(M_total.T, index=adata_state.var_names, columns=adata_state.obs_names)
    ntr_df   = pd.DataFrame(M_ntr.T,   index=adata_state.var_names, columns=adata_state.obs_names)

    to_csv_named(total_df, f"totalrna_{tag}_full_allgenes.csv")
    to_csv_named(ntr_df,   f"ntr_{tag}_full_allgenes.csv")
    print(f"Exported {state}: totalrna_{tag}_full_allgenes.csv and ntr_{tag}_full_allgenes.csv")

# Shared genes between 4sU dataset and the no-4sU external dataset
C = adata_qc_raw.layers["C"]; T = adata_qc_raw.layers["T"]
if sparse.issparse(C): C = C.toarray()
if sparse.issparse(T): T = T.toarray()

# === External NO-4sU control counts (cells × genes); transpose to cells×genes
df_no4su = pd.read_csv(
    "/Users/ersinilbay/Downloads/GSM4671630_CK-TFEA-run1n2_ds3_gene_exonic.intronic_tagged.dge.txt",
    sep="\t", index_col=0
).T

shared_genes_state = adata_qc_raw.var_names.intersection(df_no4su.columns)
gene_idx_state = adata_qc_raw.var_names.get_indexer(shared_genes_state)
total_no4su_state = df_no4su[shared_genes_state].sum(axis=0).to_numpy()

# ---------- Helper to build per-state DF of dropout vs half-life ----------
def _per_state_df(half_life_map: dict, max_hl: float = None):
    frames = []
    for state in adata_qc_raw.obs["cell_state"].cat.categories:
        mask = (adata_qc_raw.obs["cell_state"] == state).values
        # per-state totals in 4sU dataset
        tot_4su = (C[mask, :] + T[mask, :])[:, gene_idx_state].sum(axis=0)
        df = pd.DataFrame({
            "gene": list(shared_genes_state),
            "total_4su": np.asarray(tot_4su).flatten(),
            "total_no4su": np.asarray(total_no4su_state).flatten(),
        }).set_index("gene")

        # Filter to avoid zeros/low counts
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df["total_4su"] > 10) & (df["total_no4su"] > 10)]

        # map half-life source
        df["half_life_hr"] = df.index.map(half_life_map)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["half_life_hr"])
        if max_hl is not None:
            df = df[df["half_life_hr"] <= max_hl]

        # dropout metric (center within state for comparability)
        df["log2FC"] = np.log2(df["total_4su"] / df["total_no4su"])
        df["log2FC_centered"] = df["log2FC"] - df["log2FC"].median()
        df["state"] = state
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True)

# ===== Per-state dropout vs NTR rank — FRACTION normal + detection filter =====

# Dense layers
C = adata_qc_raw.layers["C"]; T = adata_qc_raw.layers["T"]
if sparse.issparse(C): C = C.toarray()
if sparse.issparse(T): T = T.toarray()

states = list(adata_umap.obs["cell_state"].cat.categories)

# Genes present in both datasets
shared = adata_qc_raw.var_names.intersection(df_no4su.columns)
idx = adata_qc_raw.var_names.get_indexer(shared)

# ---------- GLOBAL NTR rank (same x for all panels) ----------
C_all = C[:, idx]; T_all = T[:, idx]
with np.errstate(divide="ignore", invalid="ignore"):
    ntr_global_vec = np.where((C_all.sum(axis=0) + T_all.sum(axis=0)) > 0,
                              C_all.sum(axis=0) / (C_all.sum(axis=0) + T_all.sum(axis=0)),
                              np.nan)
ntr_global = pd.Series(np.asarray(ntr_global_vec).flatten(), index=shared).dropna()
shared = ntr_global.index
idx = adata_qc_raw.var_names.get_indexer(shared)
ntr_rank_global = ntr_global.rank(ascending=False, method="average")  # 1..N

# ---------- FRACTION normalisation ----------
X0 = df_no4su[shared].to_numpy()                    # cells x genes (no-4sU)
lib0 = X0.sum(axis=1, keepdims=True)
frac0 = (X0 / np.clip(lib0, 1, None)).mean(axis=0)  # mean fraction per gene
det0  = (X0 > 0).mean(axis=0)                       # detection rate in control
eps = 1e-12
frac0 = np.clip(frac0, eps, None)

# thresholds (avoid zeros and outliers)
MIN_DET = 0.02      # require ≥2% cells detected in BOTH datasets
WINSOR  = 6.0       # clip plotted log2FC_centered to [-6, 6]; set None to disable
JITTER  = 0.35      # x-jitter to break rank ties

frames, stats_rows = [], []

for state in states:
    mask = (adata_qc_raw.obs["cell_state"] == state).values
    Xs = (C[mask, :][:, idx] + T[mask, :][:, idx])      # 4sU counts for this state (cells x genes)
    libs = Xs.sum(axis=1, keepdims=True)
    fracs = (Xs / np.clip(libs, 1, None))               # per-cell fractions
    frac_s = fracs.mean(axis=0)                          # mean fraction per gene (state)
    det_s  = (Xs > 0).mean(axis=0)                       # detection rate in 4sU state

    frac_s = np.clip(np.asarray(frac_s).flatten(), eps, None)
    det_s  = np.asarray(det_s).flatten()

    # keep genes with acceptable detection in BOTH datasets
    keep = (det0 >= MIN_DET) & (det_s >= MIN_DET)
    g    = np.array(shared)[keep]
    f0   = frac0[keep]
    fs   = frac_s[keep]

    log2fc = np.log2(fs / f0)
    log2fc_centered = log2fc - np.median(log2fc)
    if WINSOR is not None:
        log2fc_centered = np.clip(log2fc_centered, -WINSOR, WINSOR)

    df = pd.DataFrame({
        "gene": g,
        "ntr_rank_order": ntr_rank_global.loc[g].values,
        "log2FC_centered": log2fc_centered,
        "state": state
    })
    frames.append(df)
    rho, p = spearmanr(df["ntr_rank_order"], df["log2FC_centered"])
    stats_rows.append((state, df.shape[0], float(np.median(log2fc_centered)), float(rho), float(p)))

df_state_diag = pd.concat(frames, ignore_index=True)
print(pd.DataFrame(stats_rows, columns=["state","n_genes","median_log2FC_centered","spearman_rho","p_value"]))



fig, axes = plt.subplots(1, len(states), figsize=(len(states)*5.6, 4.4),
                         constrained_layout=True, sharey=True)
if len(states) == 1: axes = [axes]

cmap = mpl.colormaps.get_cmap("viridis")
norm = _Normalize(vmin=0, vmax=1)
rng = np.random.default_rng(42)

for ax, state in zip(axes, states):
    sub = df_state_diag[df_state_diag["state"] == state].copy()
    x = sub["ntr_rank_order"].to_numpy(dtype=float)
    y = sub["log2FC_centered"].to_numpy(dtype=float)
    if JITTER:
        x = x + rng.uniform(-JITTER, JITTER, size=x.shape)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)

    ax.scatter(x, y, c=z, cmap=cmap, s=12, alpha=0.85, edgecolors="none")
    sns.regplot(x=x, y=y, scatter=False, lowess=True, ax=ax, line_kws={"lw": 2})

    rho = sub[["ntr_rank_order","log2FC_centered"]].corr(method="spearman").iloc[0,1]
    ax.set_title(f"{state}\nSpearman \u03C1 = {rho:.2f}")
    ax.axhline(0, ls="--", lw=1, c="gray")
    ax.set_xlabel("Gene Rank by NTR (High \u2192 Low)")

axes[0].set_ylabel("Mean-centered log\u2082 FC (4sU / no4sU)")
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=np.ravel(axes).tolist(), fraction=0.02, pad=0.02)
cbar.set_label("Local Density")
savefig_named("dropout_vs_NTRrank_by_state.svg", format="svg")
plt.show()
# ===== End fraction+det-filter per-state diagnostic =====

# ===== Per-state kinetic parameters (half-life, deg_rate, synth_rate) =====

# dense layers (local names so we don't disturb your C/T elsewhere)
C_ks = adata_qc_raw.layers["C"]; T_ks = adata_qc_raw.layers["T"]
if sparse.issparse(C_ks): C_ks = C_ks.toarray()
if sparse.issparse(T_ks): T_ks = T_ks.toarray()

genes = adata_qc_raw.var_names.to_numpy()

rows = []
for state in states:
    mask = (adata_qc_raw.obs["cell_state"] == state).values

    # pseudo-bulk sums per gene within state
    Csum = C_ks[mask, :].sum(axis=0).astype(float)
    Tsum = T_ks[mask, :].sum(axis=0).astype(float)
    Tot  = Csum + Tsum

    # stable ratio + masks
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(Csum, Tot, out=np.zeros_like(Csum), where=Tot > 0)
    ratio = np.clip(ratio, 0.0, 1.0 - 1e-12)
    valid = (Tot > 0) & (ratio > 0) & (ratio < 1.0 - 1e-12)

    # half-life (hr), deg_rate (hr^-1), synth_rate (units consistent with your global block)
    hl = np.full_like(Tot, np.nan, dtype=float)
    hl[valid] = -T_label * np.log(2) / np.log1p(-ratio[valid])

    deg = np.full_like(hl, np.nan, dtype=float)
    deg[valid] = np.log(2) / hl[valid]

    # per-cell mean labeled counts in THIS state for synth_rate
    new_mean_state = C_ks[mask, :].mean(axis=0).astype(float)
    exp_term = np.exp(-deg * T_label)
    synth = np.full_like(deg, np.nan, dtype=float)
    ok = np.isfinite(deg) & (exp_term != 1)
    synth[ok] = (new_mean_state[ok] * deg[ok]) / (1 - exp_term[ok])

    df_state = pd.DataFrame({
        "gene": genes,
        "state": state,
        "half_life_hr": hl,
        "deg_rate": deg,
        "synth_rate": synth,
        "Csum": Csum,
        "Tsum": Tsum
    })
    rows.append(df_state)

kin_by_state = pd.concat(rows, ignore_index=True)
plot_df = kin_by_state.replace([np.inf, -np.inf], np.nan).copy()
to_csv_named(kin_by_state, "gene_kinetics_by_state.csv")

def clip_pct(s, lo=0.01, hi=0.99):
    s = s.astype(float)
    return s.clip(s.quantile(lo), s.quantile(hi))

plot_df_deg  = plot_df[np.isfinite(plot_df["deg_rate"])]
plot_df_deg  = plot_df_deg.assign(deg_rate_clip=plot_df_deg.groupby("state")["deg_rate"].transform(clip_pct))

plot_df_syn  = plot_df[np.isfinite(plot_df["synth_rate"])]
plot_df_syn  = plot_df_syn.assign(synth_rate_clip=plot_df_syn.groupby("state")["synth_rate"].transform(clip_pct))

# --------- PLOTTING: paired state-vs-state scatter (Pluripotent vs 2C-like) ----------
def paired_scatter(df, param, a="Pluripotent", b="2-cell like", fname="paired.svg",
                   max_x=None):
    pvt = df.pivot_table(index="gene", columns="state", values=param, aggfunc="first")
    sub = pvt[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if param == "half_life_hr":
        sub = sub[(sub[a] > 0) & (sub[b] > 0) & (sub[a] < 24) & (sub[b] < 24)]
        lim = max_x or float(np.nanmax(sub[[a, b]].to_numpy()))
        lim = max(lim, 1.0)
    else:
        lo = sub.quantile(0.01); hi = sub.quantile(0.99)
        sub = sub.clip(lower=lo, upper=hi, axis=1)
        lim = max_x or float(np.nanmax(sub[[a, b]].to_numpy()))
    x = sub[a].to_numpy(); y = sub[b].to_numpy()
    xy = np.vstack([x, y]); z = gaussian_kde(xy)(xy)
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    cmap = mpl.colormaps.get_cmap("viridis")

    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(x, y, c=z, cmap=cmap, s=14, alpha=0.9, edgecolors="none")
    plt.plot([0, lim], [0, lim], ls="--", c="gray")
    r, p = spearmanr(x, y)
    plt.text(0.05, 0.05, f"Spearman \u03C1 = {r:.2f}", transform=plt.gca().transAxes,
             bbox=dict(facecolor="white", alpha=0.6))
    plt.xlabel(f"{a} {param.replace('_',' ')}")
    plt.ylabel(f"{b} {param.replace('_',' ')}")
    plt.title(f"{param.replace('_',' ')}: {a} vs {b}")
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    savefig_named(fname, format="svg")
    plt.show()

# =================== HVG exports (per state + all) ===================
# Identify HVG gene names from adata_umap
hvg_gene_names = adata_umap.var_names[adata_umap.var["highly_variable"]]

# Subset adata_qc_raw by filtered cells and HVG genes
adata_allhvg = adata_qc_raw[adata_umap.obs_names, hvg_gene_names].copy()

# Copy cell state annotations
adata_allhvg.obs["cell_state"] = adata_umap.obs["cell_state"].reindex(adata_allhvg.obs_names)

# Extract C and T layers and convert to dense
C = adata_allhvg.layers["C"]; T = adata_allhvg.layers["T"]
C_dense = C.toarray() if sparse.issparse(C) else C
T_dense = T.toarray() if sparse.issparse(T) else T

slug = {"Pluripotent": "pluripotent", "Intermediate": "intermediate", "2-cell like": "2cell_like"}

# Total dataset (all cells)
pd.DataFrame(C_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names)\
 .to_csv(out_path(add_suffix("newrna_hvg.csv")))
pd.DataFrame(T_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names)\
 .to_csv(out_path(add_suffix("oldrna_hvg.csv")))
print("Exported HVG-based NASC-seq2 input: newrna_hvg.csv and oldrna_hvg.csv")

# Per-state exports
for state in adata_allhvg.obs["cell_state"].cat.categories:
    mask = adata_allhvg.obs["cell_state"] == state
    selected_cells = adata_allhvg.obs_names[mask]

    C_sub = C_dense[mask, :]
    T_sub = T_dense[mask, :]

    C_df = pd.DataFrame(C_sub.T, index=adata_allhvg.var_names, columns=selected_cells)
    T_df = pd.DataFrame(T_sub.T, index=adata_allhvg.var_names, columns=selected_cells)

    tag = slug[state]
    C_df.to_csv(out_path(add_suffix(f"newrna_{tag}_hvg.csv")))
    T_df.to_csv(out_path(add_suffix(f"oldrna_{tag}_hvg.csv")))
    print(f"Exported {state}: newrna_{tag}_hvg.csv and oldrna_{tag}_hvg.csv")

# ---- Per-state FULL (all genes) exports ----
C_full = adata_qc_raw.layers["C"]
T_full = adata_qc_raw.layers["T"]
C_full_dense = C_full.toarray() if sparse.issparse(C_full) else C_full
T_full_dense = T_full.toarray() if sparse.issparse(T_full) else T_full

for state in adata_umap.obs["cell_state"].cat.categories:
    mask = (adata_umap.obs["cell_state"] == state)
    selected_cells = adata_umap.obs_names[mask]
    selected_cells = selected_cells.intersection(adata_qc_raw.obs_names)
    idx_rows = adata_qc_raw.obs_names.get_indexer(selected_cells)

    C_sub = C_full_dense[idx_rows, :]
    T_sub = T_full_dense[idx_rows, :]

    C_df_full = pd.DataFrame(C_sub.T, index=adata_qc_raw.var_names, columns=selected_cells)
    T_df_full = pd.DataFrame(T_sub.T, index=adata_qc_raw.var_names, columns=selected_cells)

    tag = slug[state]
    C_df_full.to_csv(out_path(add_suffix(f"newrna_{tag}_full.csv")))
    T_df_full.to_csv(out_path(add_suffix(f"oldrna_{tag}_full.csv")))
    print(f"Exported {state}: newrna_{tag}_full.csv and oldrna_{tag}_full.csv")

# All cells together for NASC-seq2 HVG fitting
all_C_df = pd.DataFrame(C_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names)
all_T_df = pd.DataFrame(T_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names)
all_C_df.to_csv(out_path(add_suffix("newrna_allcells_hvg.csv")))
all_T_df.to_csv(out_path(add_suffix("oldrna_allcells_hvg.csv")))
print("Exported NASC-seq2 fitting input for all cells: newrna_allcells_hvg.csv and oldrna_allcells_hvg.csv")
