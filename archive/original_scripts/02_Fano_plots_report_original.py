#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import math 
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os, re

# ---------- reproducibility ----------
np.random.seed(42)

# optional label repulsion
try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except Exception:
    _HAS_ADJUSTTEXT = False

# optional GO/Enrichr
try:
    import gseapy as gp
    _HAS_GSEAPY = True
except Exception:
    _HAS_GSEAPY = False

# ======================== USER PATHS =========================
REP1_QC = Path("/Users/ersinilbay/PycharmProjects/Master-internship/rep1_properfiltering/adata_qc_raw_with_kinetics_and_states_fix.h5ad")
REP1_UM = Path("/Users/ersinilbay/PycharmProjects/Master-internship/rep1_properfiltering/adata_umap_with_states_fix.h5ad")
REP2_QC = Path("/Users/ersinilbay/PycharmProjects/Master-internship/rep2_fix/adata_qc_raw_with_kinetics_and_states_rep2_fix.h5ad")
REP2_UM = Path("/Users/ersinilbay/PycharmProjects/Master-internship/rep2_fix/adata_umap_with_states_rep2_fix.h5ad")
OUT_DIR = Path("/Users/ersinilbay/PycharmProjects/Master-internship/fanostuff")

# ======================== FIGURE STYLE (paper-like) ==========
# roomier panels; annotated residuals a bit larger
FIGSIZE_SCAT        = (4.8, 4.8)
FIGSIZE_RESID_SMALL = (5.2, 5.2)
ANNOTATED_SCALE     = 1.15
FIGSIZE_RESID_ANN   = (FIGSIZE_RESID_SMALL[0]*ANNOTATED_SCALE,
                       FIGSIZE_RESID_SMALL[1]*ANNOTATED_SCALE)


plt.style.use("default")
plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,


    # fonts (smaller)
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,

    # axes & ticks (slightly thinner/smaller)
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,

    # legend box
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.3",
})


# ======================== STATE CONFIG =======================
STATE_MAP = {"Pluripotent": "pluri", "Intermediate": "inter", "2-cell like": "2C"}
PLOT_STATES: set[str] | None = None   # e.g. {"pluri","inter","2C"} to restrict
STATE_COL: str | None = None

# colors
BLUE  = "#1f77b4"
RIDGE = "0.25"
RED   = "crimson"

# ======================== GENE LISTS =========================
GENES_REP1 = [
    "2410141K09Rik","Abcf2","Api5","Aplp2","Arpc1a","B230219D22Rik","Ccdc59","Cdca8","Cdk9",
    "Cry1","Csnk1a1","Ddx47","Dnajc21","Eda2r","Eps15l1","Faf1","Fam168b","Farsa","Fnta","Gar1",
    "Gatad2a","Gid8","Gm42418","Gpi1","Hmmr","Hspd1","Isg20l2","Jade1","Kat6a","Keap1","Lpin1",
    "Lsm14a","Malat1","Mat2b","Mob4","mt-Nd1","mt-Nd5","Nanog","Nom1","Nus1","Paxip1","Pbk",
    "Ptbp1","Rad18","Ranbp10","Rbm25","Rfx7","Rpl3","Scpep1","Sin3b","Smn1","Spice1","Tdgf1",
    "Thumpd3","Top2a","Tra2a","Trim33","Uso1","Wbp4","Xiap","Xrcc5","Zfp106","Zfp42"
]
GENES_REP2 = [
    "Cct7","Cenpj","Copb1","Cyb5r3","Dag1","Ddx47","Dnajc5","Edrf1","Eif3a","Fam193a",
    "Gm12346","Gm26624","Gm42418","Golga7","H3f3b","Hnrnph1","Hspa14","Incenp","Jade1",
    "Map3k7","Map4","Mrps31","Mybl2","Ncor1","Nup205","Pnn","Prpf3","Prpf8","Rbbp6",
    "Rhebl1","Rpl23","Rtf1","Scpep1","Slc38a1","Smc4","Tnks","Tpt1","Txndc9","Uso1",
    "Wtap","Zfp654","Zic3"
]

SELECTED_LABELS = {
    "Pluripotent_rep1": ["Nanog","Zfp42","Tdgf1","Rpl3","Farsa","Gpi1","Cdk9"],
    "Pluripotent_rep2": ["Zic3","Mybl2","Ncor1","Eif3a","Rpl23","H3f3b"],
}

# ---- REP1 pluri overlay groups for Residual-Fano (final 4 buckets; n=51) ----
# ---- REP1 pluri overlay groups for Residual-Fano (final 4 buckets; n=51) ----
REP1_PLURI_GROUPS = {
    "Pluripotency / early-embryo signaling": [
        "Zfp42","Dppa5a","Lefty2","L1td1","Tdgf1"
    ],
    "Chromatin / genome regulation": [
        "Set","Top2a","Smc4","H3f3b","Smarca5","Hmgb2","Nasp","Nap1l1",
        "Mtf2","Pds5a"
    ],
    "RNA biology": [
        "Dqx1","Hnrnpa2b1","Hnrnpc","G3bp2","Eif2s2","Eif3a","Eif5a",
        "Eef1a1","Eef1b2","Eef2","Nop56","Nop58","Rbm25","Neat1",
        "Mrps31","Rpl23"
    ],
    "Other regulators": [
        "Ankfy1","Arl6ip1","Cbfb","Cep95","Ctsl","Dennd5b","Fkbp3","Hmmr",
        "Macrod2","Malat1","Mdm2","Nedd4","Nek1","Npl","Nucks1","Pdgfrl",
        "Peg10","Psma7","Ran","Samd4b","Sgsm3","mt-Nd5"
    ],
}
REP1_GROUP_COLORS  = {
    "Pluripotency / early-embryo signaling": "#2ca02c",
    "Chromatin / genome regulation":         "#ff7f0e",
    "RNA biology":                            "#9467bd",
    "Other regulators":                       "crimson",
}
REP1_GROUP_MARKERS = {
    "Pluripotency / early-embryo signaling": "o",
    "Chromatin / genome regulation":         "s",
    "RNA biology":                           "D",
    "Other regulators":                      "o",
}


# ---- REP2 pluri overlay groups for Residual-Fano (4 buckets) ----
REP2_PLURI_GROUPS = {
    "Pluripotency / early-embryo signaling": [
        "Zfp42","Dppa5a","Lefty2","L1td1","Tdgf1"
    ],
    "Chromatin / genome regulation": [
        "Top2a","Smc4","H3f3b","Mtf2","Nap1l1","Hmgb2",
        "Cenpf","Pcna","Ube2c","Mdm2"
    ],
    "RNA biology": [
        "Dqx1","Hnrnpa2b1","Hnrnpc","Hnrnpm","Hnrnpab","G3bp2",
        "Eif3a","Eif2s2","Eif5a","Eef1a1","Eef1b2","Eef2",
        "Nop56","Nop58","Rbm25","Neat1","Rsl1d1","Pcbp2","Cct6a","Rpl23"
    ],
    "Other regulators": [
        "Fkbp3","Psma7","Hsp90b1","Hspd1","Nedd4","Nucks1","Npl","Ran",
        "Pdgfrl","Peg10","mt-Nd5",
        "Ldha","Pkm","Atp5b","Cox4i1","Slc25a5","Gpx1","Prdx1","Glrx2","Mt1","Mt2",
        "Vim","Tpm1","Tpm3","Tuba1b","Add1","Sparc","Fn1","Cald1",
        "Apoe","Fabp3","Nefl","Etfdh","Ddit4","Pmaip1","Calm1"
    ],
}
# reuse the same palette/markers as REP1
REP2_GROUP_COLORS  = REP1_GROUP_COLORS
REP2_GROUP_MARKERS = REP1_GROUP_MARKERS

# ===== Showcase panel gene sets =====
PLURI_FOCUS = [
    "Nanog","Zfp42","Tdgf1","Lefty2","Dppa5a","Klf4","Nodal","Top2a"
]  # add/remove as you like

RNA_FOCUS = [
    "Hnrnpc","Hnrnpa2b1","Srsf3","Nsun2","Dnajc21","Eif3a","Hspa5","Hsp90b1","Calr"
]

PRC_FOCUS = [
    "Ezh2","Suz12","Mtf2","Jarid2","Rbbp4","Rbbp7","Rnf2","Pcgf2","Pcgf4",
    "Cbx7","Cbx8","Atrx","Kdm2b","Nsd1"
]

GROUP_PALETTE = {"pluri": "#2ca02c", "rna": "#9467bd", "prc": "#ff7f0e"}
GROUP_MARK    = {"pluri": "o",       "rna": "D",       "prc": "s"}



# ======================== RESIDUAL-FANO CONFIG ===============
RESIDUAL_CFG = dict(
    # how many labels to show on the annotated plot
    n_label=35,

    # basic filters
    min_mean=0.1,
    max_mean=6.0,                # allow higher-mean genes to appear

    # --- Selection rule ---
    select_rule="zscore",        # "zscore" | "fold" | "fano_global"

    # STRICT thresholds (used for enrichment and “official” outlier set)
    z_thr=1.2,
    min_fano=2.0,

    z_thr_display=0.9,
    min_fano_display=1.6,

    # alternative modes (kept for completeness)
    top_frac_global=0.10,        # only if select_rule="fano_global"
    min_fold=1.5,                # only if select_rule="fold"

    # exclusions + detection
    # exclusions + detection
    exclude_pref=("mt-", "Rpl", "Rps","Mrps", "Gm"),
    exclude_suf=("Rik", "-ps"),
    whitelist=(),  # names to *always* keep despite the excludes
    min_detect_frac=0.10,

    # fitting expected log-Fano
    winsor_q=0.99,
)


# Per-state overrides — keeps 2C clean on the left
RESIDUAL_STATE_OVERRIDES = {
    "pluri": {
        "n_label": 40,          # cap labels if you ever turn labels on
        "z_thr_display": 4.2,
        "min_fano_display": 1.80,
        "z_thr": 4.5,
        "min_fano": 1.80,

        "whitelist": ("Rpl23", "mt-Nd5"),
        "min_mean": 0.07,
        "max_mean": 6.0
    },
    "2C": {
        "n_label": 40,
        "z_thr_display": 1.1,
        "min_fano_display": 2.0
    },
}






PRIORITY_2C = ["Zscan4","Zscan4c","Zscan4d","Zscan4f","Zscan4-ps1","Zscan4-ps2","Zscan4-ps3"]
PRIORITY_LENIENT = dict(min_mean=1e-3, max_mean=5.0, min_fold=1.7)

# ===================== ENRICHMENT CONFIG (global) ====================
ENRICH_STRICT = False
PREFERRED_ENRICH_LIBRARIES = ("GO_Biological_Process_2021",)

# Collect Supplementary tables across runs; written once at the end
SUPP_TABLES: list[tuple[str, pd.DataFrame]] = []

# Backward-compat alias
ENRICH_LIBRARIES = PREFERRED_ENRICH_LIBRARIES

# Optional local GMT
CHROMATIN_GMT = None

PRINT_ENRICHR_CATALOG = True

# GO plotting & filtering
GO_MIN_OVERLAP = 3
GO_MAX_TERMS   = 10
GO_FDR_MAX     = 0.05

# keep GO bar axes comparable across figures
GO_FIG_WIDTH_IN  = 6.0
GO_ROW_HEIGHT_IN = 0.34
GO_LEFT_MARGIN   = 0.46
GO_BAR_HEIGHT    = 0.65
GO_TERM_MAXCHARS = 55

# global numeric x-limit used by all GO plots in this run (largest −log10 p observed)
GO_GLOBAL_XMAX = 0.0

# ======================== I/O HELPERS ========================
def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def _save_current_fig(out_dir: Path, filename: str) -> None:
    _ensure_outdir(out_dir)
    try:
        plt.gcf().canvas.draw()
        plt.savefig(out_dir / filename, format="svg", bbox_inches="tight")
    except Exception as e:
        print(f"[WARN] SVG save failed for {filename}: {e} — trying PNG fallback.")
        plt.savefig(out_dir / (Path(filename).with_suffix(".png")), dpi=300)

# ======================== MATRIX HELPERS =====================
def _preferred_layer(adata: sc.AnnData, names: list[str]) -> np.ndarray | None:
    for nm in names:
        if nm in adata.layers:
            X = adata.layers[nm]
            return X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return None

def _fraction_new_from_layers(adata: sc.AnnData) -> np.ndarray | None:
    NEW = _preferred_layer(adata, ["new", "C", "new_counts", "newrna"])
    TOT = _preferred_layer(adata, ["total", "T", "total_counts", "oldrna_plus_newrna", "counts"])
    if NEW is None or TOT is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = NEW / np.maximum(TOT, 1e-12)
        frac[~np.isfinite(frac)] = 0.0
    return frac

def _new_fraction_matrix(adata_qc: sc.AnnData) -> np.ndarray:
    frac = _preferred_layer(adata_qc, ["new_frac", "ntr", "NTR"])
    if frac is not None:
        return frac
    frac = _fraction_new_from_layers(adata_qc)
    if frac is not None:
        return frac
    raise ValueError("Couldn't find NEW fraction. Provide 'new_frac' (or NEW/TOTAL layers).")

def _total_matrix(adata_qc: sc.AnnData) -> np.ndarray:
    X = _preferred_layer(adata_qc, ["total", "T", "total_counts", "counts"])
    if X is not None:
        return X
    X = adata_qc.X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def _safe_stats_over_cells(mat: np.ndarray, mask_cells: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if mask_cells is not None:
        mat = mat[mask_cells, :]
    if mat.size == 0:
        n_genes = mat.shape[1] if mat.ndim == 2 else 0
        return np.full(n_genes, np.nan), np.full(n_genes, np.nan)
    mat = np.asarray(mat, dtype=np.float64)
    mean = np.nanmean(mat, axis=0)
    var  = np.nanvar(mat,  axis=0, ddof=1)
    return np.asarray(mean).ravel(), np.asarray(var).ravel()

def _detect_fraction(TOT: np.ndarray, mask_cells: np.ndarray | None) -> np.ndarray:
    X = TOT[mask_cells, :] if mask_cells is not None else TOT
    if hasattr(X, "toarray"):
        X = X.toarray()
    return (X > 0).mean(axis=0).ravel()

def _present_cols(df: pd.DataFrame) -> list[str]:
    """Keep only columns that exist (robust across gseapy versions)."""
    return [c for c in ["Term","n","Adjusted P-value","log10p","Overlap","Genes"] if c in df.columns]

# ======================== STATE HELPERS ======================
def detect_state_column(ad_um: sc.AnnData) -> str | None:
    if STATE_COL and STATE_COL in ad_um.obs:
        return STATE_COL
    candidates_priority = [
        "state","State","umap_state","UMAP_state","cell_state","CellState",
        "annotation","annotations","label","labels","celltype","CellType",
        "cell_type","leiden","louvain","cluster","clusters"
    ]
    for c in candidates_priority:
        if c in ad_um.obs:
            return c
    hits = []
    for c in ad_um.obs.columns:
        cname = str(c).lower()
        if ("state" in cname) or ("annot" in cname) or ("cluster" in cname) or ("celltype" in cname):
            hits.append(c)
    return hits[0] if hits else None

def _canonical_state_code(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    # direct mapping
    if val in STATE_MAP:
        return STATE_MAP[val]
    s = str(val)
    # already code?
    for pretty, code in STATE_MAP.items():
        if s == code:
            return code
    # normalized text equals pretty?
    s_norm = s.strip().lower().replace(" ", "").replace("-", "")
    for pretty, code in STATE_MAP.items():
        p_norm = pretty.strip().lower().replace(" ", "").replace("-", "")
        if s_norm == p_norm:
            return code
    return s

def _sync_state(ad_qc: sc.AnnData, ad_um: sc.AnnData) -> None:
    col = detect_state_column(ad_um)
    if col is None:
        print("[WARN] Could not detect a state column in UM.")
        preview = [(c, ", ".join(ad_um.obs[c].astype(str).unique()[:5])) for c in ad_um.obs.columns[:20]]
        print(pd.DataFrame(preview, columns=["obs column", "sample values"]).to_string(index=False))
        return
    shared = ad_qc.obs_names.intersection(ad_um.obs_names)
    if len(shared) == 0:
        print("[WARN] No shared cell barcodes between QC and UM.")
        return
    ad_qc.obs.loc[shared, "state"] = ad_um.obs.loc[shared, col].values
    ad_qc.obs["state"] = ad_qc.obs["state"].map(_canonical_state_code)
    vc = ad_qc.obs["state"].value_counts(dropna=False)
    print(f"[INFO] Using UM obs['{col}'] as state; normalized counts:\n{vc}")

def _mask_for_state(adata: sc.AnnData, state_code: str) -> np.ndarray | None:
    if "state" not in adata.obs:
        print("[WARN] obs['state'] not present; using ALL cells.")
        return None
    m = (adata.obs["state"] == state_code).values
    if m.sum() == 0:
        print(f"[WARN] No cells in state '{state_code}'.")
        return None
    return m

# ======================== PLOTTING HELPERS ===================
def _prep_ax(ax, logx=True, logy=True, grid=True, square_axes=False):
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if grid:
        ax.yaxis.grid(True, which="major", color="#E0E0E0", alpha=0.6, lw=0.8)
        ax.xaxis.grid(False)
    for sp in ax.spines.values():
        sp.set_alpha(0.9)
        sp.set_linewidth(1.2)
    # NOTE: no forced square aspect; constrained layout will size things correctly.


# --- PMFs for Poisson / NegBin overlays on count histograms ---
def _pois_pmf(k, mu):
    k = np.asarray(k, dtype=float)
    return np.exp(k*np.log(mu + 1e-12) - mu - np.vectorize(math.lgamma)(k + 1.0))

def _nbinom_pmf_from_mean_var(k, mu, var):
    """Parameterize NB from mean/variance: var = mu + mu^2/r  ->  r = mu^2/(var - mu)."""
    if var <= mu + 1e-12:
        return _pois_pmf(k, mu)
    r = (mu*mu) / (var - mu)          # 'size'
    p = r / (r + mu)                  # success prob
    k = np.asarray(k, dtype=float)
    return np.exp(
        np.vectorize(math.lgamma)(k + r) - np.vectorize(math.lgamma)(r) - np.vectorize(math.lgamma)(k + 1.0)
        + r*np.log(p) + k*np.log(1.0 - p)
    )

def _annotate_points(ax, x, y, names, fontsize=5.8):
    """
    Labels with collision-avoidance. Draw a short, straight connector
    ONLY if the label moved > ~10 px from its dot (keeps things neat).
    """
    texts = []
    for xi, yi, nm in zip(x, y, names):
        t = ax.text(
            xi, yi, nm,
            fontsize=fontsize, color="black", alpha=0.98,
            ha="left", va="bottom", clip_on=False,
            path_effects=[pe.Stroke(linewidth=1.4, foreground="white", alpha=0.95), pe.Normal()],
            bbox=dict(facecolor="white", alpha=0.55, pad=0.30, edgecolor="none"),
        )
        texts.append(t)

    # repel labels (no arrows here)
    if _HAS_ADJUSTTEXT:
        adjust_text(
            texts, x=x, y=y, ax=ax,
            expand_points=(2.6, 2.8),
            expand_text=(1.5, 1.5),
            force_text=(0.9, 0.9),
            force_points=(0.55, 0.55),
            only_move={'points': 'y', 'text': 'xy'},
            autoalign='y',
            avoid_self=True,
            lim=2800,
            precision=0.001
        )

    # straight connectors only if the label actually moved (pixel threshold)
    for t, xi, yi in zip(texts, x, y):
        tx, ty = t.get_position()
        p1 = ax.transData.transform((xi, yi))
        p2 = ax.transData.transform((tx, ty))
        dx, dy = (p2 - p1)
        if (dx*dx + dy*dy) ** 0.5 >= 10:  # 10 px threshold
            # shorten a bit so the line doesn't run into the label box
            frac = 0.9
            px = xi + (tx - xi) * frac
            py = yi + (ty - yi) * frac
            ax.plot([xi, px], [yi, py],
                    lw=0.8, alpha=0.9, color="0.25", solid_capstyle="round")

    return texts


def _panel_caption(ax, *, n_genes: int, min_mean: float, min_detect_frac: float,
                   n_cells: int, rule_text: str):
    det_cells = math.ceil(min_detect_frac * max(n_cells, 1))
    txt = (f"N={n_genes} genes | filter: mean ≥ {min_mean:g}, detected ≥ {det_cells} cells "
           f"(~{100*min_detect_frac:.0f}%) | {rule_text}")
    ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=7, color="0.3")

# ======================== CORE PLOTS =========================

def plot_mean_variance(tag: str, mean: np.ndarray, var: np.ndarray, where: str, state_tag: str, out_dir: Path):
    m = np.isfinite(mean) & np.isfinite(var) & (mean > 0) & (var > 0)
    x, y = mean[m], var[m]
    plt.figure(figsize=FIGSIZE_SCAT); ax = plt.gca(); _prep_ax(ax, True, True)
    ax.scatter(x, y, s=9, color=BLUE, alpha=0.45, edgecolors="none")
    ax.set_title(f"{tag} {where}: mean–variance ({state_tag})")
    ax.set_xlabel("Mean"); ax.set_ylabel("Variance")
    plt.tight_layout(pad=0.7); _save_current_fig(out_dir, f"{tag}_{state_tag}_{where}_mean-variance.svg"); plt.show()


def plot_fano_vs_mean(tag: str, mean: np.ndarray, var: np.ndarray,
                      where: str, state_tag: str, out_dir: Path):
    """Fano (Var/Mean) vs Mean scatter with Poisson line."""
    m = np.isfinite(mean) & np.isfinite(var) & (mean > 0)
    x = mean[m]; fano = var[m] / np.maximum(x, 1e-20)
    plt.figure(figsize=FIGSIZE_SCAT); ax = plt.gca(); _prep_ax(ax, True, True)
    ax.scatter(x, fano, s=9, color=BLUE, alpha=0.45, edgecolors="none")
    ax.axhline(1.0, ls=":", lw=1.1, color=BLUE, alpha=0.9, label="Poisson Fano = 1")
    ax.set_title(f"{tag} {where}: Fano vs mean ({state_tag})")
    ax.set_xlabel("Mean"); ax.set_ylabel("Fano = Var/Mean")
    ax.legend(loc="upper left", framealpha=0.92)
    plt.tight_layout(pad=0.7); _save_current_fig(out_dir, f"{tag}_{state_tag}_{where}_fano-vs-mean.svg"); plt.show()


def plot_global_mean_variance(tag: str, ad_qc: sc.AnnData, out_dir: Path):
    """Wrapper used by run_one(). Plots TOTAL mean–variance for the current ad_qc."""
    state_tag = "all"
    if "state" in ad_qc.obs and ad_qc.obs["state"].nunique() == 1:
        state_tag = str(ad_qc.obs["state"].unique()[0])
    TOT = _total_matrix(ad_qc)
    mean, var = _safe_stats_over_cells(TOT, mask_cells=None)
    plot_mean_variance(tag, mean, var, where="TOTAL", state_tag=state_tag, out_dir=out_dir)


def gene_stats_from_total(adata: sc.AnnData) -> pd.DataFrame:
    """Per-gene mean/variance/Fano from TOTAL layer (or X)."""
    X = _total_matrix(adata)
    mean, var = _safe_stats_over_cells(X, mask_cells=None)
    fano = var / np.maximum(mean, 1e-20)
    return pd.DataFrame({"mean": mean, "variance": var, "fano": fano}, index=adata.var_names)


def list_offscale_fano_genes(adata: sc.AnnData, *, fano_min: float = 10.0, mean_min: float = 0.10,
                             out_csv: Path | None = None, tag: str = "") -> list[str]:
    """List genes with very large Fano at non-tiny mean; optionally write CSV."""
    st = gene_stats_from_total(adata)
    keep = (st["fano"] >= fano_min) & (st["mean"] >= mean_min) & np.isfinite(st["fano"])
    df = st.loc[keep].sort_values(["fano", "mean"], ascending=[False, False])
    if out_csv is not None:
        _ensure_outdir(out_csv.parent)
        df.to_csv(out_csv)
        print(f"[INFO] {tag}: saved off-scale Fano genes -> {out_csv} (n={len(df)})")
    else:
        print(f"[INFO] {tag}: off-scale Fano genes n={len(df)})")
    return df.index.tolist()


def _plot_one_panel_groups(ax, adata_pluri: sc.AnnData, layer_name: str, title: str,
                           groups: dict[str, list[str]]):
    """Single panel background cloud + highlighted groups (by TOTAL)."""
    X = _total_matrix(adata_pluri)
    mean, var = _safe_stats_over_cells(X)
    m = np.isfinite(mean) & np.isfinite(var) & (mean > 0)
    x = mean[m]; fano = var[m] / np.maximum(x, 1e-20); names = np.array(adata_pluri.var_names)[m]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(x, fano, s=12, color=BLUE, alpha=0.28, edgecolors="none")
    name_to_i = {g: i for i, g in enumerate(names)}
    for gname, glist in groups.items():
        present = [g for g in glist if g in name_to_i]
        if not present:
            continue
        gx = np.array([x[name_to_i[g]] for g in present])
        gy = np.array([fano[name_to_i[g]] for g in present])
        ax.scatter(gx, gy, s=62, edgecolors="black", linewidths=0.55, alpha=0.98, label=f"{gname} (n={len(present)})")
        lbl = present[:min(12, len(present))]
        _annotate_points(ax,
                         np.array([x[name_to_i[g]] for g in lbl]),
                         np.array([fano[name_to_i[g]] for g in lbl]),
                         lbl, fontsize=6.0)
    ax.axhline(1.0, ls=":", lw=1.1, color=BLUE, alpha=0.9)
    ax.set_title(title); ax.set_xlabel("Mean (TOTAL)"); ax.set_ylabel("Fano = Var/Mean")
    ax.legend(loc="upper left", framealpha=0.92)


def plot_gene_distributions_two_reps(adata_left: sc.AnnData, adata_right: sc.AnnData,
                                     gene_list: list[str],
                                     stats_left: pd.DataFrame, stats_right: pd.DataFrame,
                                     label_left: str, label_right: str,
                                     title: str, log_scale: bool = False,
                                     save_svg: bool = True, save_dir: Path | None = None,
                                     show_fits: str | None = "nb"):
    """
    Side-by-side panels (one per gene). For each gene: overlay rep1 vs rep2 histograms
    of TOTAL counts and draw optional NegBin/Poisson PMF fits from stats tables.
    Style: rep1=green (filled), rep2=blue (step), NB fits solid lines.
    """
    Xl = _total_matrix(adata_left)
    Xr = _total_matrix(adata_right)
    idx_l = {g: i for i, g in enumerate(adata_left.var_names)}
    idx_r = {g: i for i, g in enumerate(adata_right.var_names)}

    # colors
    COL_L = "#2ca02c"  # green
    COL_R = "#1f77b4"  # blue

    n = max(1, len(gene_list))
    fig_w = 5.2 * min(3, n)
    fig, axs = plt.subplots(1, n, figsize=(fig_w, 3.8), squeeze=False)
    axs = axs.ravel()

    for ax, g in zip(axs, gene_list):
        if g not in idx_l or g not in idx_r:
            ax.set_visible(False)
            print(f"[WARN] distributions: {g} not in both datasets; skipping.")
            continue

        vl = np.asarray(Xl[:, idx_l[g]]).ravel()
        vr = np.asarray(Xr[:, idx_r[g]]).ravel()
        vmax = int(max(vl.max() if vl.size else 0, vr.max() if vr.size else 0))
        bins = np.arange(0, max(1, vmax) + 2)

        # histograms (density so PMF overlays on same scale)
        ax.hist(vl, bins=bins, density=True, alpha=0.55, label=label_left,
                color=COL_L, edgecolor="none")
        ax.hist(vr, bins=bins, density=True, histtype="step", linewidth=1.2,
                label=label_right, color=COL_R)

        # optional fits from stats tables
        if show_fits:
            def _draw_fit(stats_df, color, lab):
                if g not in stats_df.index:
                    return
                mu = float(stats_df.at[g, "mean"])
                var = float(stats_df.at[g, "variance"])
                k = np.arange(0, max(1, vmax) + 1, dtype=float)
                if show_fits in ("nb", "both"):
                    y = _nbinom_pmf_from_mean_var(k, mu, var)
                    ax.plot(k, y, lw=1.4, color=color, alpha=0.95, label=f"NB fit ({lab})")
                if show_fits in ("poisson", "both"):
                    y = _pois_pmf(k, mu)
                    ax.plot(k, y, lw=1.2, color=color, alpha=0.75, ls="--", label=f"Pois fit ({lab})")

            _draw_fit(stats_left,  COL_L, label_left)
            _draw_fit(stats_right, COL_R, label_right)

        if log_scale:
            ax.set_yscale("log")

        ax.set_title(g, fontsize=10)
        ax.set_xlabel("counts"); ax.set_ylabel("probability density")
        ax.legend(framealpha=0.92, fontsize=8)

    fig.suptitle(title, y=1.02, fontsize=11)
    plt.tight_layout(pad=0.9)
    if save_svg and save_dir is not None:
        _ensure_outdir(save_dir)
        _save_current_fig(save_dir, "distributions_two_reps.svg")
    plt.show()



def plot_distributions_2x2_overlay(ad1: sc.AnnData, ad2: sc.AnnData,
                                   genes_for_page: list[str],
                                   stats1: pd.DataFrame, stats2: pd.DataFrame,
                                   out_path: Path, show_fits: str | None = "nb"):
    """
    2×2 grid, each subplot overlays rep1 (green, filled) vs rep2 (blue, step) for one gene.
    Draw optional NegBin/Poisson fits using per-gene mean/variance.
    """
    X1 = _total_matrix(ad1); X2 = _total_matrix(ad2)
    i1 = {g: i for i, g in enumerate(ad1.var_names)}
    i2 = {g: i for i, g in enumerate(ad2.var_names)}
    genes = genes_for_page[:4]
    if not genes:
        print("[INFO] plot_distributions_2x2_overlay: nothing to plot.");
        return

    COL_L = "#2ca02c"; COL_R = "#1f77b4"

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.6))
    axs = axs.ravel()

    for ax, g in zip(axs, genes):
        if g not in i1 or g not in i2:
            ax.set_visible(False);
            continue
        v1 = np.asarray(X1[:, i1[g]]).ravel()
        v2 = np.asarray(X2[:, i2[g]]).ravel()
        vmax = int(max(v1.max() if v1.size else 0, v2.max() if v2.size else 0))
        bins = np.arange(0, max(1, vmax) + 2)

        ax.hist(v1, bins=bins, density=True, alpha=0.55, color=COL_L, edgecolor="none", label="rep1")
        ax.hist(v2, bins=bins, density=True, histtype="step", linewidth=1.2, color=COL_R, label="rep2")

        if show_fits:
            k = np.arange(0, max(1, vmax) + 1, dtype=float)
            if g in stats1.index:
                mu, var = float(stats1.at[g, "mean"]), float(stats1.at[g, "variance"])
                if show_fits in ("nb", "both"):
                    ax.plot(k, _nbinom_pmf_from_mean_var(k, mu, var), color=COL_L, lw=1.2, label="NB fit (rep1)")
                if show_fits in ("poisson", "both"):
                    ax.plot(k, _pois_pmf(k, mu), color=COL_L, lw=1.0, ls="--", label="Pois fit (rep1)")
            if g in stats2.index:
                mu, var = float(stats2.at[g, "mean"]), float(stats2.at[g, "variance"])
                if show_fits in ("nb", "both"):
                    ax.plot(k, _nbinom_pmf_from_mean_var(k, mu, var), color=COL_R, lw=1.2, label="NB fit (rep2)")
                if show_fits in ("poisson", "both"):
                    ax.plot(k, _pois_pmf(k, mu), color=COL_R, lw=1.0, ls="--", label="Pois fit (rep2)")

        ax.set_title(g)
        ax.set_xlabel("counts"); ax.set_ylabel("probability density")
        ax.legend(framealpha=0.92, fontsize=8)

    for ax in axs[len(genes):]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.9)
    _ensure_outdir(out_path.parent)
    plt.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    plt.show()
    print(f"[INFO] Saved: {out_path}")



# ======================== Expected-Fano ======================
def _fit_expected_logfano(log_mean: np.ndarray, log_fano: np.ndarray,
                          nbins: int = 40, smooth: int = 3, min_per_bin: int = 30):
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(log_mean, qs)
    edges = np.unique(edges)
    if edges.size < 5:
        edges = np.linspace(log_mean.min(), log_mean.max(), 20)
    bins = np.digitize(log_mean, edges[1:-1], right=False)
    med_x, med_y = [], []
    for b in range(bins.min(), bins.max() + 1):
        m = bins == b
        if m.sum() >= min_per_bin:
            med_x.append(np.median(log_mean[m]))
            med_y.append(np.median(log_fano[m]))
    med_x = np.asarray(med_x); med_y = np.asarray(med_y)
    if med_x.size == 0:
        med_x = np.array([np.median(log_mean)-1e-6, np.median(log_mean)+1e-6])
        med_y = np.array([np.median(log_fano), np.median(log_fano)])
    if smooth > 1 and med_y.size >= smooth:
        k = np.ones(smooth) / smooth
        med_y = np.convolve(med_y, k, mode="same")
    order = np.argsort(med_x); med_x, med_y = med_x[order], med_y[order]
    def evaluate(logm: np.ndarray) -> np.ndarray:
        return np.interp(logm, med_x, med_y, left=med_y[0], right=med_y[-1])
    return med_x, med_y, evaluate

# ======================== Residual-Fano core =================
def _residual_prepare_arrays(adata_qc: sc.AnnData, mask_cells, state_tag: str):
    cfg = RESIDUAL_CFG.copy()
    ov  = RESIDUAL_STATE_OVERRIDES.get(state_tag, {})
    for k, v in ov.items():
        cfg[k] = v

    TOT = _total_matrix(adata_qc)
    mean, var = _safe_stats_over_cells(TOT, mask_cells=mask_cells)
    det_all = _detect_fraction(TOT, mask_cells)

    m = np.isfinite(mean) & np.isfinite(var) & (mean > 0)
    x = mean[m]; f = var[m] / np.maximum(mean[m], 1e-20)
    det = det_all[m]
    names = np.array(adata_qc.var_names)[m].astype(str)

    # base keep: mean/detect only
    keep_base = (x >= cfg["min_mean"]) & (x <= cfg["max_mean"])
    n_cells_state = (mask_cells.sum() if isinstance(mask_cells, np.ndarray) else adata_qc.n_obs)
    eff_min_det_frac = max(cfg.get("min_detect_frac", 0.0), 2.0 / max(n_cells_state, 1))
    keep_base &= (det >= eff_min_det_frac)

    # prefix/suffix excludes with whitelist override
    keep_excl = np.ones_like(keep_base, dtype=bool)
    for pref in tuple(cfg.get("exclude_pref", ())):
        keep_excl &= ~np.char.startswith(names, pref)
    for suf in tuple(cfg.get("exclude_suf", ())):
        keep_excl &= ~np.char.endswith(names, suf)

    wl = set(map(str, cfg.get("whitelist", ())))
    if wl:
        keep_excl |= np.isin(names, list(wl))  # re-include whitelisted genes only

    keep = keep_base & keep_excl

    x, f, names, det = x[keep], f[keep], names[keep], det[keep]
    return x, f, names, det, cfg

def _expected_curve_eval(x: np.ndarray, f: np.ndarray, winsor_q: float):
    logx = np.log10(x); logf = np.log10(f)
    if 0.9 < winsor_q < 1.0:
        q = np.quantile(logf, winsor_q)
        logf_fit = np.clip(logf, None, q)
    else:
        logf_fit = logf
    _, _, eval_fn = _fit_expected_logfano(logx, logf_fit, nbins=40, smooth=3, min_per_bin=30)
    exp_f = np.power(10.0, eval_fn(logx))
    return eval_fn, exp_f

# ======================== Residual-Fano plots =================
def plot_fano_total_residual_outliers(
    tag: str,
    adata_qc: sc.AnnData,
    mask_cells: np.ndarray | None,
    state_tag: str,
    out_dir: Path,
    overlay_groups: dict[str, list[str]] | None = None,
    overlay_colors: dict[str, str] | None = None,
    overlay_markers: dict[str, str] | None = None,
    *,
    show_cutoff: bool = True,        # draw the display cutoff curve
    show_strict_only: bool = False,  # if True: red dots = STRICT outliers (smaller set)
    dump_tables: bool = True         # print + save CSVs of genes above cutoff
):
    """
    Residual-Fano scatter for TOTAL layer. Uses state-specific overrides from
    RESIDUAL_STATE_OVERRIDES, draws the display cutoff curve, and (optionally)
    prints/saves tables of genes above the cutoff.
    Returns: list of STRICT outlier genes (used for enrichment).
    """
    # ---- prep arrays & filters
    x, f, names, det, cfg = _residual_prepare_arrays(adata_qc, mask_cells, state_tag)
    if len(f) == 0:
        print(f"[WARN] {tag}-{state_tag}: no genes pass residual filters.")
        return []

    # ---- expected curve & residuals
    _, exp_f = _expected_curve_eval(x, f, cfg["winsor_q"])
    resid_ratio = f / np.maximum(exp_f, 1e-20)

    logx   = np.log10(x)
    logf   = np.log10(f)
    logexp = np.log10(exp_f)
    resid  = logf - logexp                                  # log-Fano residual
    sigma  = 1.4826 * np.median(np.abs(resid - np.median(resid))) or 1e-9  # robust SD (MAD)

    # ---- thresholds (strict vs display)
    rule             = cfg.get("select_rule", "zscore")
    z_thr            = float(cfg.get("z_thr", 1.2))
    min_fano         = float(cfg.get("min_fano", 2.0))
    z_thr_display    = float(cfg.get("z_thr_display", z_thr))
    min_fano_display = float(cfg.get("min_fano_display", min_fano))
    top_frac_global  = cfg.get("top_frac_global", None)
    min_fold         = cfg.get("min_fold", None)

    if rule == "zscore":
        z = resid / sigma
        strict_mask = (z >= z_thr)         & (f >= min_fano)
        loose_mask  = (z >= z_thr_display) & (f >= min_fano_display)
        score       = z
    elif rule == "fold":
        thr         = float(min_fold if isinstance(min_fold, (int, float)) and np.isfinite(min_fold) else 1.5)
        strict_mask = (resid_ratio >= thr) & (f >= min_fano)
        loose_mask  = (resid_ratio >= max(1.3, thr - 0.2)) & (f >= min_fano_display)
        score       = resid_ratio
    elif rule == "fano_global":
        qthr        = float(np.quantile(f, 1.0 - float(top_frac_global or 0.10)))
        strict_mask = (f >= qthr)
        loose_mask  = (f >= np.quantile(f, 1.0 - float(top_frac_global or 0.10) * 1.3))
        score       = f
    else:
        raise ValueError(f"Unknown select_rule: {rule}")

    if not np.any(loose_mask):
        print(f"[WARN] {tag}-{state_tag}: no display outliers under current settings.")
        return []

    # ---- choose which mask to *display* (reds)
    display_mask = strict_mask if show_strict_only else loose_mask

    # ---- cutoff curve for the display z-threshold (in Fano units)
    # F_cut(x) = expected_fano(x) * 10^(z_thr_display * sigma)
    f_cut = exp_f * (10.0 ** (z_thr_display * sigma))

    # ---- figure
    plt.figure(figsize=FIGSIZE_RESID_ANN)  # annotated = big
    ax = plt.gca()
    _prep_ax(ax, True, True)

    # background cloud
    ax.scatter(x, f, s=12, color=BLUE, alpha=0.28, edgecolors="none")

    # draw cutoff curve
    # draw cutoff line, but do NOT put it in the legend (declutters)
    if show_cutoff:
        order = np.argsort(x)
        ax.plot(
            x[order], f_cut[order],
            color="0.35", lw=1.2, ls="--", label="_nolegend_"
        )

    # panel naming + label mode
    title_extra = ""
    suffix      = ""
    do_label    = True
    if overlay_groups and len(overlay_groups) == 1:
        gname = next(iter(overlay_groups))
        safe = re.sub(r'[^A-Za-z0-9]+', '_', gname).strip('_')
        title_extra = f" — {gname}"
        suffix = f"_{safe}"
        do_label = True  # label the genes of THIS group

    # red points = displayed outliers
    x_o, f_o, names_o = x[display_mask], f[display_mask], names[display_mask]
    score_o           = score[display_mask]
    resid_ratio_o     = resid_ratio[display_mask]

    order = np.argsort(score_o)[::-1]
    sel_idx_sub = order[:len(order)]  # label ALL displayed outliers
    sel_x, sel_f, sel_n = x_o[sel_idx_sub], f_o[sel_idx_sub], names_o[sel_idx_sub]

    # overlay grouping (only among displayed outliers)
    h_groups = {}
    is_group = np.zeros_like(sel_idx_sub, dtype=bool)
    if overlay_groups:
        name_to_i_full = {n.lower(): i for i, n in enumerate(names_o)}
        for grp, glist in overlay_groups.items():
            _ids = [name_to_i_full[g.lower()] for g in glist if g.lower() in name_to_i_full]
            if not _ids:
                continue
            gx, gy = x_o[_ids], f_o[_ids]
            h_groups[grp] = ax.scatter(
                gx, gy, s=58,
                marker=(overlay_markers or {}).get(grp, "o"),
                color=(overlay_colors or {}).get(grp, "#2ca02c"),
                edgecolors="black", linewidths=0.55, alpha=0.98,
                label=f"{grp} (n={len(_ids)})"
            )
            # mark which selected outliers belong to a group (for red/other split)
            sel_names_set = {n for n in names_o[_ids]}
            is_group |= np.array([n in sel_names_set for n in sel_n])

    # non-group displayed outliers (red)
    h_other = ax.scatter(sel_x[~is_group], sel_f[~is_group], s=24, color=RED, alpha=0.96,
                         edgecolors="black", linewidths=0.45,
                         label=f"Other high-Fano outliers (n={(~is_group).sum()})")

    # labels: restrict to overlay group members (and explicit whitelist) to avoid clutter
    if do_label:
        overlay_set = set()
        if overlay_groups:
            for glist in overlay_groups.values():
                overlay_set.update(g.lower() for g in glist)

        # also allow labels for config whitelist genes
        wl = set(map(str.lower, cfg.get("whitelist", ())))
        allow = np.array([(nm.lower() in overlay_set) or (nm.lower() in wl) for nm in sel_n], dtype=bool)

        # if nothing qualifies (e.g., no groups present), fall back to top-N by score
        if not allow.any():
            k = min(int(cfg.get("n_label", 35)), len(sel_n))
            allow = np.zeros_like(allow, dtype=bool)
            allow[order[:k]] = True

        _annotate_points(
            ax,
            sel_x[allow],
            sel_f[allow],
            sel_n[allow],
            fontsize=(5.8 if allow.sum() <= 22 else 5.6)
        )


    ax.set_title(f"{tag} TOTAL: Residual-Fano outliers ({state_tag}){title_extra}")
    ax.set_xlabel("Mean (TOTAL)")
    ax.set_ylabel("Fano = Var/Mean")

    # legend (groups + "other" only)
    legend_items = list(h_groups.values()) + [h_other]
    ax.legend(legend_items, [h.get_label() for h in legend_items],
              loc="upper left", framealpha=0.92, borderaxespad=0.2, ncol=1)

    plt.tight_layout(pad=0.9)
    _save_current_fig(out_dir, f"{tag}_{state_tag}_TOTAL_fano-vs-mean_residual-outliers{suffix}.svg")
    plt.show()

    # ---------- dump tables (console + CSV) ----------
    if dump_tables:
        # recompute z only if needed
        z = resid / sigma if rule == "zscore" else score
        disp_mask = (z >= z_thr_display) & (f >= min_fano_display) if rule == "zscore" else display_mask
        core_mask = (z >= z_thr)         & (f >= min_fano)         if rule == "zscore" else strict_mask

        table = pd.DataFrame({
            "gene":  names,
            "mean":  x,
            "fano":  f,
            "expected_fano": exp_f,
            "z" : (resid / sigma) if rule == "zscore" else np.nan,
            "resid_log10": resid,
            "above_display_cut": disp_mask,
            "strict_core": core_mask,
        }).sort_values(["above_display_cut","strict_core","z","fano"], ascending=[False,False,False,False])

        to_print = table.loc[table["above_display_cut"], ["gene","mean","fano","z"]]
        print(
            f"\n=== genes above display cutoff (sorted by z) — display={int(disp_mask.sum())}, strict={int(core_mask.sum())} ===")
        # cap print size to keep console sane
        with pd.option_context("display.max_rows", 600, "display.width", 140):
            print(to_print.to_string(index=False))

        out_all  = out_dir / f"{tag}_{state_tag}_residual_fano_table.csv"
        out_disp = out_dir / f"{tag}_{state_tag}_residual_fano_display_only.csv"
        table.to_csv(out_all, index=False)
        to_print.to_csv(out_disp, index=False)
        print(f"[INFO] Saved: {out_all}")
        print(f"[INFO] Saved: {out_disp}")

    # return STRICT outliers (for enrichment)
    return list(names[strict_mask])


def _residual_arrays_and_masks(adata_qc, mask_cells, state_tag):
    # same selection logic used in residual panel
    x, f, names, det, cfg = _residual_prepare_arrays(adata_qc, mask_cells, state_tag)
    _, exp_f = _expected_curve_eval(x, f, cfg["winsor_q"])
    resid_ratio = f / np.maximum(exp_f, 1e-20)

    logx   = np.log10(x)
    logf   = np.log10(f)
    logexp = np.log10(exp_f)
    resid  = logf - logexp
    sigma  = 1.4826 * np.median(np.abs(resid - np.median(resid))) or 1e-9
    z      = resid / sigma

    rule             = cfg.get("select_rule", "zscore")
    z_thr            = float(cfg.get("z_thr", 1.5))
    min_fano         = float(cfg.get("min_fano", 2.5))
    z_thr_display    = float(cfg.get("z_thr_display", z_thr))
    min_fano_display = float(cfg.get("min_fano_display", min_fano))

    if rule != "zscore":
        # if you ever flip to "fold", adapt here; we keep zscore for the showcase
        pass

    strict_mask = (z >= z_thr)         & (f >= min_fano)
    loose_mask  = (z >= z_thr_display) & (f >= min_fano_display)

    return x, f, names, exp_f, strict_mask, loose_mask, cfg

def make_residual_groups_2x2(
    tag: str,
    adata_qc: sc.AnnData,
    state_tag: str,
    out_dir: Path,
    groups: dict[str, list[str]],
    colors: dict[str, str],
    markers: dict[str, str],
):
    x, f, names, exp_f, strict_mask, loose_mask, cfg = _residual_arrays_and_masks(
        adata_qc, mask_cells=None, state_tag=state_tag
    )
    x_all, f_all, names_all = x, f, names
    display_mask = loose_mask

    # cutoff handle
    _, exp_f_full = _expected_curve_eval(x_all, f_all, cfg["winsor_q"])
    resid = np.log10(f_all) - np.log10(exp_f_full)
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) or 1e-9
    z_thr_display = float(cfg.get("z_thr_display", cfg.get("z_thr", 1.2)))
    f_cut = exp_f_full * (10.0 ** (z_thr_display * sigma))

    def panel(ax, group_name: str):
        _prep_ax(ax, True, True)
        ax.scatter(x_all, f_all, s=12, color=BLUE, alpha=0.28, edgecolors="none")
        order = np.argsort(x_all)
        ax.plot(
            x_all[order], f_cut[order],
            color="0.35", lw=1.2, ls="--",
            label="_nolegend_"
        )
        # displayed outliers for context
        x_disp = x_all[display_mask]; f_disp = f_all[display_mask]; n_disp = names_all[display_mask]
        h_other = ax.scatter(x_disp, f_disp, s=18, color="lightcoral", alpha=0.85,
                             edgecolors="black", linewidths=0.35, label="Other high-Fano outliers")

        # group ∩ display
        want = set(groups.get(group_name, []))
        sel = np.array([g in want for g in n_disp])
        # group ∩ display
        n_in_group = 0
        if sel.any():
            gx, gy, gn = x_disp[sel], f_disp[sel], n_disp[sel]
            ax.scatter(
                gx, gy, s=62,
                color=colors.get(group_name, "black"),
                marker=markers.get(group_name, "o"),
                edgecolors="black", linewidths=0.55, alpha=0.98,
            )
            _annotate_points(ax, gx, gy, gn, fontsize=5.8 if sel.sum() <= 22 else 5.6)
            n_in_group = int(sel.sum())

        # titles only (no legend), include n=
        ax.set_title(f"{group_name} (n={n_in_group})", fontsize=10)
        ax.set_xlabel("Mean (TOTAL)");
        ax.set_ylabel("Fano = Var/Mean")

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.2))
    order_names = [
        "Pluripotency / early-embryo signaling",
        "Chromatin / genome regulation",
        "RNA biology",
        "Other regulators",
    ]
    for ax, nm in zip(axs.ravel(), order_names):
        panel(ax, nm)
    fig.suptitle(f"{tag} TOTAL: Residual-Fano outliers ({state_tag}) — grouped", y=0.995, fontsize=11)
    plt.tight_layout(pad=1.0)
    outp = OUT_DIR / f"{tag}_{state_tag}_TOTAL_residual_groups_2x2.svg"
    fig.savefig(outp, format="svg", bbox_inches="tight", dpi=300)
    plt.show(); plt.close(fig)
    print(f"[INFO] Saved: {outp}")

# ======================== ENRICHMENT HELPERS =================

def _enrichr_available_libraries() -> set[str]:
    if not _HAS_GSEAPY:
        return set()
    try:
        return set(gp.get_library_name())
    except Exception:
        try:
            return set(gp.get_libraries())
        except Exception:
            return set()

def _validate_requested_libraries(requested: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return (valid, missing) Enrichr library names for this environment."""
    if not _HAS_GSEAPY:
        raise RuntimeError("gseapy is not installed. Run: pip install -U gseapy pandas")
    available = _enrichr_available_libraries()
    valid   = [nm for nm in requested if nm in available]
    missing = [nm for nm in requested if nm not in available]
    return valid, missing

def _run_one_enrichr(gene_list: list[str], universe: list[str], library) -> pd.DataFrame:
    """Run enrichr for a single library (string name or dict). Returns tidy df or empty."""
    try:
        if isinstance(library, dict):
            enr = gp.enrichr(gene_list=gene_list, gene_sets=library,
                             background=universe, outdir=None, cutoff=1.0, verbose=False)
        else:
            enr = gp.enrichr(gene_list=gene_list, gene_sets=library,
                             background=universe, outdir=None, cutoff=1.0, verbose=False)
        if enr.results is None or enr.results.empty:
            return pd.DataFrame()
        df = enr.results.copy()

        # Normalize column names
        for cand in ["Adjusted P-value", "Adjusted P value", "Adj P-value", "Adjusted Pval", "FDR q-value", "FDR"]:
            if cand in df.columns:
                df.rename(columns={cand: "Adjusted P-value"}, inplace=True)
                break

        if "Genes" not in df.columns:
            for cand in ["Gene_set", "Gene Set", "GeneSet", "genes"]:
                if cand in df.columns:
                    df.rename(columns={cand: "Genes"}, inplace=True)
                    break

        if "Overlap" not in df.columns and "Genes" in df.columns:
            def _make_overlap_from_genes(s):
                if pd.isna(s):
                    return "0/0"
                n = len([g for g in str(s).replace(";", ",").split(",") if g.strip()])
                return f"{n}/?"
            df["Overlap"] = df["Genes"].map(_make_overlap_from_genes)

        keep_cols = [c for c in ["Term", "Adjusted P-value", "Overlap", "Genes"] if c in df.columns]
        df = df[keep_cols]
        return df
    except Exception as e:
        print(f"[WARN] Enrichr failed for '{library}': {e}")
        return pd.DataFrame()

def _standardize_enrich_df_full(df: pd.DataFrame, fdr_max: float) -> pd.DataFrame:
    """Tidy full Enrichr result for Supplementary: add n + -log10p; FDR filter; sort."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    # normalize adj p column
    if "Adjusted P-value" not in d.columns:
        for cand in ["FDR q-value", "FDR", "P.adjust", "adjp", "q-value", "qvalue"]:
            if cand in d.columns:
                d.rename(columns={cand: "Adjusted P-value"}, inplace=True)
                break
    if "Adjusted P-value" not in d.columns:
        return pd.DataFrame()

    # n-overlap
    if "Overlap" in d.columns:
        def _n_from_overlap(s):
            try:
                a, _ = str(s).split("/")
                return int(a)
            except Exception:
                return np.nan
        d["n"] = d["Overlap"].map(_n_from_overlap)
    elif "Genes" in d.columns:
        def _n_from_genes(s):
            if pd.isna(s): return 0
            return len([g for g in str(s).replace(";", ",").split(",") if g.strip()])
        d["n"] = d["Genes"].map(_n_from_genes)
    else:
        d["n"] = np.nan

    d = d[d["Adjusted P-value"] <= fdr_max].copy()
    if d.empty:
        return d

    d["log10p"] = -np.log10(d["Adjusted P-value"].clip(lower=1e-300))
    cols = _present_cols(d)
    d = d[cols].sort_values(["log10p","n"], ascending=[False,False])
    return d

def _collect_supp_table(sheet_name: str, df: pd.DataFrame):
    """Store Supplementary sheet (Excel sheet names max 31 chars)."""
    if df is None or df.empty:
        return
    safe = sheet_name.replace("/", "_").replace(" ", "_")
    if len(safe) > 31:
        safe = safe[:31]
    SUPP_TABLES.append((safe, df.copy()))

def _max_log10p_from_df(d: pd.DataFrame, fdr_max: float) -> float:
    """Return the max -log10(adj p) in a results df after FDR filtering; 0.0 if NA."""
    if d is None or d.empty:
        return 0.0
    d = d.copy()
    if "Adjusted P-value" not in d.columns:
        for cand in ["FDR q-value", "FDR", "P.adjust", "adjp", "q-value", "qvalue"]:
            if cand in d.columns:
                d.rename(columns={cand: "Adjusted P-value"}, inplace=True)
                break
    if "Adjusted P-value" not in d.columns:
        return 0.0
    d = d[d["Adjusted P-value"] <= fdr_max]
    if d.empty:
        return 0.0
    return float((-np.log10(d["Adjusted P-value"].clip(lower=1e-300))).max())

def _plot_go_bar(
    df: pd.DataFrame, title: str, out_path: Path,
    max_terms: int = GO_MAX_TERMS, fdr_max: float = GO_FDR_MAX,
    figsize: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    bar_height: float = GO_BAR_HEIGHT,
    show_xgrid: bool = False,
    box_all_spines: bool = True
) -> pd.DataFrame:
    if df.empty:
        print(f"[INFO] No enriched terms for: {title}")
        return pd.DataFrame()

    df = df.copy()

    # --- derive n-overlap ---
    def _olap_n_from_overlap(s):
        try:
            a, _ = str(s).split("/")
            return int(a)
        except Exception:
            return np.nan
    n_vals = df["Overlap"].map(_olap_n_from_overlap) if "Overlap" in df.columns else None
    if (n_vals is None) or (np.isnan(n_vals).all()):
        def _count_genes(s):
            if pd.isna(s): return 0
            return len([g for g in str(s).replace(";", ",").split(",") if g.strip()])
        n_vals = df["Genes"].map(_count_genes) if "Genes" in df.columns else pd.Series([0]*len(df), index=df.index)
    df["n"] = n_vals

    # --- normalize adj p column ---
    if "Adjusted P-value" not in df.columns:
        for cand in ["FDR q-value", "FDR", "P.adjust", "adjp", "q-value", "qvalue"]:
            if cand in df.columns:
                df.rename(columns={cand: "Adjusted P-value"}, inplace=True)
                break
    if "Adjusted P-value" not in df.columns:
        print(f"[INFO] No 'Adjusted P-value' column for: {title}")
        return pd.DataFrame()

    # --- filter & score ---
    df = df[df["Adjusted P-value"] <= fdr_max]
    if df.empty:
        print(f"[INFO] All terms filtered by FDR for: {title}")
        return pd.DataFrame()
    df["log10p"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
    df = df.sort_values(["log10p", "n"], ascending=[False, False]).head(max_terms).iloc[::-1]

    # --- figure sizing ---
    n = len(df)
    if figsize is None:
        fig_w = GO_FIG_WIDTH_IN
        base_h = max(1.6, n * GO_ROW_HEIGHT_IN + 0.8)
        fig_h = 0.8 if n == 1 else base_h   # half-height for single term
    else:
        fig_w, fig_h = figsize

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))  # ensures 'ax' is defined

    # --- labels (truncate so width is stable) ---
    def _shorten(s, k=GO_TERM_MAXCHARS):
        s = str(s)
        return s if len(s) <= k else s[:k-1] + "…"
    ylbls = [f"{_shorten(t)} (n={int(nv)})" for t, nv in zip(df["Term"], df["n"].fillna(0))]

    # --- bars ---
    y = np.arange(n)
    bar_h_local = (0.45 if n == 1 else bar_height)  # slimmer bar for tiny panel
    ax.barh(y, df["log10p"], height=bar_h_local)
    ax.set_yticks(y)
    ax.set_yticklabels(ylbls, fontsize=9)
    ax.set_xlabel(r"$-\log_{10}$ (adj $p$)", fontsize=9)
    ax.set_title(title, fontsize=11)

    # --- consistent x-range ---
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmax = float(np.nanmax(df["log10p"])) if len(df) else 1.0
        ax.set_xlim(0.0, xmax * 1.05)

    # --- styling ---
    ax.spines["top"].set_visible(box_all_spines)
    ax.spines["right"].set_visible(box_all_spines)
    ax.grid(axis="x", color="0.92", visible=show_xgrid)
    ax.margins(x=0.02)
    plt.subplots_adjust(left=GO_LEFT_MARGIN, right=0.98, top=0.90, bottom=0.12)

    _save_current_fig(out_path.parent, out_path.name)
    plt.show()
    return df

# ==================== NEW: Pluri regime combo GO =============
# Your regime gene lists (as given)
PLURI_REP1_BOTTOMRIGHT = [
    "2410141K09Rik","Abcf2","Api5","Aplp2","Arpc1a","B230219D22Rik","Ccdc59","Cdca8","Cdk9","Ddx47","Eps15l1","Faf1",
    "Fam168b","Farsa","Fnta","Gar1","Gatad2a","Gid8","Gpi1","Isg20l2","Keap1","Lpin1","Lsm14a","Mat2b",
    "Mob4","Nom1","Nus1","Paxip1","Ptbp1","Ranbp10","Rfx7","Rpl3","Scpep1","Sin3b","Smn1","Thumpd3",
    "Uso1","Wbp4","Xiap"
]
PLURI_REP1_TOPLEFT = [
    "Cry1","Csnk1a1","Dnajc21","Eda2r","Gm42418","Hmmr","Hspd1","Jade1","Kat6a","Malat1","mt-Nd1","mt-Nd5",
    "Nanog","Rad18","Rbm25","Spice1","Tdgf1","Top2a","Tra2a","Trim33","Xrcc5","Zfp106","Zfp42"
]
PLURI_REP2_BOTTOMRIGHT = [
    "Cct7","Cenpj","Copb1","Cyb5r3","Dag1","Ddx47","Dnajc5","Edrf1","Fam193a","Gm12346","Golga7","Hspa14",
    "Map3k7","Map4","Mybl2","Nup205","Prpf3","Prpf8","Rhebl1","Rpl23","Rtf1","Scpep1","Slc38a1","Tnks",
    "Tpt1","Txndc9","Uso1","Wtap","Zfp654","Zic3"
]
PLURI_REP2_TOPLEFT = [
    "Eif3a","Gm26624","Gm42418","H3f3b","Hnrnph1","Incenp","Jade1","Mrps31","Ncor1","Pnn","Rbbp6","Smc4"
]

# ===== Intuitive-group overlay (no GO) =====
GROUP_COLORS = {"highFreq_lowSize": "crimson", "lowFreq_highSize": "purple"}
GROUP_MARKERS = {"highFreq_lowSize": "o",       "lowFreq_highSize": "D"}

def run_pluri_regime_combo_enrichments(out_dir: Path):
    """Combine the two regimes across both pluri reps and run GO BP enrichment for each."""
    if not _HAS_GSEAPY:
        print("[INFO] Skipping pluri-regime GO: gseapy not installed.")
        return

    universe = None  # let Enrichr decide
    highfreq_lowsize = sorted(set(PLURI_REP1_BOTTOMRIGHT) | set(PLURI_REP2_BOTTOMRIGHT))
    lowfreq_highsize = sorted(set(PLURI_REP1_TOPLEFT)      | set(PLURI_REP2_TOPLEFT))

    valid_libs, missing_libs = _validate_requested_libraries(tuple(PREFERRED_ENRICH_LIBRARIES))
    if missing_libs:
        msg = "[WARN] Missing Enrichr libraries on this server:\n" + "\n".join(f"  - {m}" for m in missing_libs)
        if ENRICH_STRICT:
            raise RuntimeError(msg + "\nSet ENRICH_STRICT=False to auto-skip missing libraries.")
        else:
            print(msg + "\nProceeding with available libraries only.")

    fdr_max = GO_FDR_MAX
    tag = "pluri_combo"

    for lib in valid_libs:
        # run both enrichments
        df_high_raw = _run_one_enrichr(highfreq_lowsize, universe, lib)
        df_low_raw  = _run_one_enrichr(lowfreq_highsize,  universe, lib)
        if (df_high_raw is None or df_high_raw.empty) and (df_low_raw is None or df_low_raw.empty):
            continue

        # Supplementary (full) tables
        _collect_supp_table(f"{tag}_highFreq_lowSize_{lib}",
                            _standardize_enrich_df_full(df_high_raw, fdr_max))
        _collect_supp_table(f"{tag}_lowFreq_highSize_{lib}",
                            _standardize_enrich_df_full(df_low_raw,  fdr_max))

        # axis sync
        global GO_GLOBAL_XMAX
        pair_max = max(_max_log10p_from_df(df_high_raw, fdr_max),
                       _max_log10p_from_df(df_low_raw,  fdr_max))
        GO_GLOBAL_XMAX = max(GO_GLOBAL_XMAX, pair_max)

        # plots + top-term CSVs
        lib_title = str(lib)

        # highFreq_lowSize
        state_tag = "highFreq_lowSize"
        title = f"{tag} {state_tag}: GO/Pathway — {lib_title}"
        outp  = OUT_DIR / f"{tag}_GO_{state_tag}_{lib_title}.svg"
        if df_high_raw is not None and not df_high_raw.empty:
            df_plot = _plot_go_bar(
                df_high_raw, title, outp,
                max_terms=GO_MAX_TERMS, fdr_max=fdr_max,
                xlim=(0.0, GO_GLOBAL_XMAX),
                show_xgrid=False, box_all_spines=True
            )
            if isinstance(df_plot, pd.DataFrame) and not df_plot.empty:
                safe_lib = lib_title.replace(" ", "_")
                csv_path = OUT_DIR / f"{tag}_{state_tag}_{safe_lib}_GO_table.csv"
                cols = _present_cols(df_plot)
                df_plot[cols].to_csv(csv_path, index=False)
                print(f"[INFO] Saved genes table (top terms): {csv_path}")

        # lowFreq_highSize
        state_tag = "lowFreq_highSize"
        title = f"{tag} {state_tag}: GO/Pathway — {lib_title}"
        outp  = OUT_DIR / f"{tag}_GO_{state_tag}_{lib_title}.svg"
        if df_low_raw is not None and not df_low_raw.empty:
            df_plot = _plot_go_bar(
                df_low_raw, title, outp,
                max_terms=GO_MAX_TERMS, fdr_max=fdr_max,
                xlim=(0.0, GO_GLOBAL_XMAX),
                show_xgrid=False, box_all_spines=True
            )
            if isinstance(df_plot, pd.DataFrame) and not df_plot.empty:
                safe_lib = lib_title.replace(" ", "_")
                csv_path = OUT_DIR / f"{tag}_{state_tag}_{safe_lib}_GO_table.csv"
                cols = _present_cols(df_plot)
                df_plot[cols].to_csv(csv_path, index=False)
                print(f"[INFO] Saved genes table (top terms): {csv_path}")



# ======================== MODEL-STABLE OVERLAY ===============
def plot_fano_total_with_highlight_labels(tag: str, adata_qc: sc.AnnData,
                                          highlight_all: list[str], label_only: list[str],
                                          mask_cells: np.ndarray | None, state_tag: str, out_dir: Path):
    TOT = _total_matrix(adata_qc)
    mean, var = _safe_stats_over_cells(TOT, mask_cells=mask_cells)

    m = np.isfinite(mean) & np.isfinite(var) & (mean > 0)
    x_all = mean[m]; f_all = var[m] / np.maximum(mean[m], 1e-20)
    names_all = np.array(adata_qc.var_names)[m]
    idx = {g: i for i, g in enumerate(names_all)}

    present = [g for g in highlight_all if g in idx]
    bx = np.array([x_all[idx[g]] for g in present]) if present else np.array([])
    by = np.array([f_all[idx[g]] for g in present]) if present else np.array([])

    label_present = [g for g in label_only if g in idx]
    lx = np.array([x_all[idx[g]] for g in label_present]) if label_present else np.array([])
    ly = np.array([f_all[idx[g]] for g in label_present]) if label_present else np.array([])

    extra_names = []
    if state_tag == "pluri" and len(present) > 0:
        scores = np.log10(np.maximum(bx, 1e-12)) + np.log10(np.maximum(by, 1e-12))
        topk = np.argsort(scores)[-3:]
        extra_names = [present[i] for i in topk if present[i] not in label_present]

    all_label_names = label_present + [g for g in extra_names if g not in label_present]
    if len(all_label_names) > 0:
        lx = np.array([x_all[idx[g]] for g in all_label_names])
        ly = np.array([f_all[idx[g]] for g in all_label_names])

    # annotated overlay should also be the bigger size (it has labels)
    plt.figure(figsize=FIGSIZE_RESID_ANN)
    ax = plt.gca(); _prep_ax(ax, True, True)
    ax.scatter(x_all, f_all, s=16, color=BLUE, alpha=0.33, edgecolors="none")
    ax.axhline(1.0, ls=":", lw=1.1, color=BLUE, alpha=0.9, label="Poisson Fano = 1")

    if len(present) > 0:
        ax.scatter(bx, by, s=40, color=RED, alpha=0.96,
                   edgecolors="black", linewidths=0.45, label="Model-stable list")

    if len(all_label_names) > 0:
        _annotate_points(ax, lx, ly, all_label_names, fontsize=8.5 if len(all_label_names)<=22 else 8.0)

    ax.set_title(f"{tag} TOTAL: Fano vs mean ({state_tag}; model-stable)")
    ax.set_xlabel("Mean (TOTAL)"); ax.set_ylabel("Fano = Var/Mean")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)
    fname = f"{tag}_{state_tag}_TOTAL_fano-vs-mean_model-stable.svg"
    plt.tight_layout(pad=0.9); _save_current_fig(out_dir, fname); plt.show()

def print_fano_summary(genes: list[str], stats1: pd.DataFrame, stats2: pd.DataFrame, thr: float = 1.5):
    rows = []
    for g in genes:
        def row(st, rep):
            if g in st.index:
                m = float(st.at[g, "mean"]); v = float(st.at[g, "variance"]); f = float(st.at[g, "fano"])
                return [g, rep, m, v, f, ("noisy" if f > thr else "near-Poisson")]
            return [g, rep, np.nan, np.nan, np.nan, "NA"]
        rows += [row(stats1, "rep1"), row(stats2, "rep2")]
    df = pd.DataFrame(rows, columns=["gene","rep","mean","variance","fano","label"])
    with pd.option_context("display.precision", 3):
        print("\n=== Per-gene Fano summary ==="); print(df)
    return df

DIST_GENES_PLURI = ["Tdgf1", "Farsa"]  # safe mESC picks in both reps

# ======================== DRIVER =============================
def run_one(tag: str, path_qc: Path, path_um: Path,
            highlight_all: list[str], label_only: list[str], out_dir: Path):
    if not (path_qc.exists() and path_um.exists()):
        print(f"[WARN] Skipping {tag}: paths not found.")
        return

    print(f"[INFO] Loading {tag}…")
    ad_qc = sc.read_h5ad(path_qc)
    ad_um = sc.read_h5ad(path_um)

    _sync_state(ad_qc, ad_um)

    # ---- FORCE PLURI-ONLY ANALYSIS ----
    m_pluri = _mask_for_state(ad_qc, "pluri")
    if m_pluri is None or m_pluri.sum() == 0:
        print(f"[ERROR] {tag}: no pluripotent cells found; aborting.")
        return
    ad_qc = ad_qc[m_pluri, :].copy()
    ad_qc.obs["state"] = "pluri"  # normalize label

    # Global (all cells) mean–variance for TOTAL (now pluri-only because we subsetted)
    plot_global_mean_variance(tag, ad_qc, out_dir)

    # We already subset to pluri above; run exactly once
    state_tag = "pluri"
    mask = None  # ad_qc contains only pluri cells

    # NEW layer
    NEW = _new_fraction_matrix(ad_qc)
    new_mean, new_var = _safe_stats_over_cells(NEW, mask_cells=mask)
    plot_mean_variance(tag, new_mean, new_var, where="NEW", state_tag=state_tag, out_dir=out_dir)
    plot_fano_vs_mean(tag, new_mean, new_var, where="NEW", state_tag=state_tag, out_dir=out_dir)

    # TOTAL layer
    TOT = _total_matrix(ad_qc)
    tot_mean, tot_var = _safe_stats_over_cells(TOT, mask_cells=mask)
    plot_mean_variance(tag, tot_mean, tot_var, where="TOTAL", state_tag=state_tag, out_dir=out_dir)
    plot_fano_vs_mean(tag, tot_mean, tot_var, where="TOTAL", state_tag=state_tag, out_dir=out_dir)

    # Residual-Fano panels
    if (tag == "rep1") and (state_tag == "pluri"):
        # (A) grouped overlay (rep1)
        genes_out = plot_fano_total_residual_outliers(
            tag, ad_qc, mask, state_tag, out_dir,
            overlay_groups=REP1_PLURI_GROUPS,
            overlay_colors=REP1_GROUP_COLORS,
            overlay_markers=REP1_GROUP_MARKERS
        )
        # (B) 2×2 grouped page (rep1)
        make_residual_groups_2x2(
            tag, ad_qc, state_tag, out_dir,
            groups=REP1_PLURI_GROUPS,
            colors=REP1_GROUP_COLORS,
            markers=REP1_GROUP_MARKERS
        )

    elif (tag == "rep2") and (state_tag == "pluri"):
        # (A) grouped overlay (rep2)
        genes_out = plot_fano_total_residual_outliers(
            tag, ad_qc, mask, state_tag, out_dir,
            overlay_groups=REP2_PLURI_GROUPS,
            overlay_colors=REP2_GROUP_COLORS,
            overlay_markers=REP2_GROUP_MARKERS
        )
        # (B) 2×2 grouped page (rep2)
        make_residual_groups_2x2(
            tag, ad_qc, state_tag, out_dir,
            groups=REP2_PLURI_GROUPS,
            colors=REP2_GROUP_COLORS,
            markers=REP2_GROUP_MARKERS
        )
        # (C) highlight your GENES_REP2 list *only if they pass display cutoff*
        #     (drawn as a separate, single-group overlay so they get labels)
        plot_fano_total_residual_outliers(
            tag, ad_qc, mask, state_tag, out_dir,
            overlay_groups={"Model-stable (rep2)": GENES_REP2},
            overlay_colors={"Model-stable (rep2)": "#333333"},
            overlay_markers={"Model-stable (rep2)": "o"},
            show_cutoff=True, show_strict_only=False, dump_tables=False
        )
        # print a quick text summary of which from GENES_REP2 passed the display cutoff
        try:
            disp_csv = out_dir / f"{tag}_{state_tag}_residual_fano_display_only.csv"
            disp_tbl = pd.read_csv(disp_csv)
            overlap = sorted(set(GENES_REP2).intersection(set(disp_tbl["gene"])))
            print(f"[INFO] {tag}-{state_tag}: {len(overlap)} of {len(GENES_REP2)} pass display cutoff:\n  " + ", ".join(
                overlap))
        except Exception as e:
            print(f"[WARN] Could not summarize GENES_REP2 overlap: {e}")

    else:
        genes_out = plot_fano_total_residual_outliers(tag, ad_qc, mask, state_tag, out_dir)

    # Model-stable overlay (pluri only) — annotated = big
    if state_tag == "pluri":
        plot_fano_total_with_highlight_labels(tag, ad_qc,
                                              highlight_all=highlight_all,
                                              label_only=label_only,
                                              mask_cells=mask,
                                              state_tag=state_tag,
                                              out_dir=out_dir)

def main():
    global GO_GLOBAL_XMAX
    GO_GLOBAL_XMAX = 0.0   # reset for this run
    _ensure_outdir(OUT_DIR)

    # ---- reps individually (unchanged) ----
    run_one("rep1", REP1_QC, REP1_UM,
            highlight_all=GENES_REP1,
            label_only=SELECTED_LABELS.get("Pluripotent_rep1", []),
            out_dir=OUT_DIR)
    run_one("rep2", REP2_QC, REP2_UM,
            highlight_all=GENES_REP2,
            label_only=SELECTED_LABELS.get("Pluripotent_rep2", []),
            out_dir=OUT_DIR)

    # ---- NEW: Pluripotent regime combo enrichment (both reps pooled) ----
    run_pluri_regime_combo_enrichments(OUT_DIR)

    # ---- Distribution insets: pluripotent cells, rep1 vs rep2 ----
    # Load once here so we can put the two reps side-by-side.
    ad1 = sc.read_h5ad(REP1_QC); um1 = sc.read_h5ad(REP1_UM); _sync_state(ad1, um1)
    ad2 = sc.read_h5ad(REP2_QC); um2 = sc.read_h5ad(REP2_UM); _sync_state(ad2, um2)

    m1 = _mask_for_state(ad1, "pluri")
    m2 = _mask_for_state(ad2, "pluri")

    # subset to pluri cells
    ad1_pluri = ad1[m1, :].copy() if m1 is not None else ad1
    ad2_pluri = ad2[m2, :].copy() if m2 is not None else ad2


    # Fano stats on TOTAL (same basis as your residual-Fano plots)
    stats1 = gene_stats_from_total(ad1_pluri)
    stats2 = gene_stats_from_total(ad2_pluri)

    _ensure_outdir(OUT_DIR)
    _ = list_offscale_fano_genes(ad1_pluri, fano_min=10.0, mean_min=0.10,
                                 out_csv=OUT_DIR / "rep1_pluri_offscale_fano.csv", tag="rep1")
    _ = list_offscale_fano_genes(ad2_pluri, fano_min=10.0, mean_min=0.10,
                                 out_csv=OUT_DIR / "rep2_pluri_offscale_fano.csv", tag="rep2")

    print_fano_summary(["Tdgf1", "Farsa", "Gpi1"], stats1, stats2)

    # NB/ZI vs Poisson showcase (your report figure)
    plot_gene_distributions_two_reps(
        adata_left=ad1_pluri,
        adata_right=ad2_pluri,
        gene_list=["Tdgf1", "Farsa"],  # noisy vs quiet
        stats_left=stats1,
        stats_right=stats2,
        label_left="rep1 pluri",
        label_right="rep2 pluri",
        title="Distributions (pluri-only): Tdgf1 (green) vs Farsa (blue)",
        save_svg=True,
        save_dir=OUT_DIR,
        show_fits="nb"  # draw NegBin only
    )

    # --- Intuitive groups (rep2 emphasis; rename keys to the labels you want on the legend) ---
    groups_r2 = {
        "pluri": PLURI_REP2_BOTTOMRIGHT,  # replace with your pluri list if different
        "chromatin": PLURI_REP2_TOPLEFT,  # replace with your chromatin list
    }
    # keep a 2×2 overview as well (rep1 uses the same two buckets for reference)
    groups_r1 = {"pluri": PLURI_REP1_BOTTOMRIGHT, "chromatin": PLURI_REP1_TOPLEFT}

    # NEW: single-panel figure focusing on REP2 with group annotations
    fig, ax = plt.subplots(figsize=(FIGSIZE_SCAT[0] * 1.3, FIGSIZE_SCAT[1] * 1.3))
    _plot_one_panel_groups(ax, ad2_pluri, "TOTAL", "rep2 pluri — TOTAL (group overlay)", groups_r2)
    plt.tight_layout(pad=0.9)
    plt.savefig(OUT_DIR / "rep2_pluri_TOTAL_groups.svg", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)

    # keep genes present in both reps
    genes_show = [g for g in DIST_GENES_PLURI if (g in ad1_pluri.var_names) and (g in ad2_pluri.var_names)]

    genes_for_page = genes_show[:4] if len(genes_show) >= 4 else genes_show
    if genes_for_page:
        plot_distributions_2x2_overlay(
            ad1_pluri, ad2_pluri, genes_for_page, stats1, stats2,
            OUT_DIR / "pluri_distributions_2x2.svg"
        )
    else:
        print(f"[INFO] Skipping distributions 2×2 — no common genes in {DIST_GENES_PLURI}.")

    if genes_show:
        plot_gene_distributions_two_reps(
            adata_left=ad1_pluri,
            adata_right=ad2_pluri,
            gene_list=genes_show,
            stats_left=stats1,
            stats_right=stats2,
            label_left="rep1 pluri",
            label_right="rep2 pluri",
            title="Pluripotent RNA distributions — key genes (rep1 vs rep2)",
            log_scale=False,
            save_svg=True,
            save_dir=OUT_DIR
        )
    else:
        missing = [g for g in DIST_GENES_PLURI if g not in ad1_pluri.var_names or g not in ad2_pluri.var_names]
        print(
            f"[INFO] Skipping distribution panel — none of {DIST_GENES_PLURI} present in BOTH reps. Missing: {missing}")

    # ---- write combined Supplementary workbook ----
    if SUPP_TABLES:
        xlsx_path = OUT_DIR / "GO_supplementary_tables.xlsx"
        with pd.ExcelWriter(xlsx_path) as xw:
            for sheet_name, df in SUPP_TABLES:
                if df is not None and not df.empty:
                    df.to_excel(xw, index=False, sheet_name=sheet_name)
        print(f"[INFO] Supplementary workbook saved: {xlsx_path}")

if __name__ == "__main__":
    main()
