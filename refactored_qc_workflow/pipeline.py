import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from config import (
    C2C_CUTOFF,
    INTER_CUTOFF,
    MARKERS,
    PLURI_MARKERS,
    STATE_CATEGORIES,
    STATE_SLUG,
    T_LABEL_HOURS,
    TARGET_PROPORTIONS,
    TWOC_MARKERS,
)


def build_initial_adata(c_df: pd.DataFrame, t_df: pd.DataFrame) -> sc.AnnData:
    C = c_df.values.T
    T = t_df.values.T
    total = C + T

    adata = sc.AnnData(
        X=total.copy(),
        var=pd.DataFrame(index=c_df.index),
        obs=pd.DataFrame(index=c_df.columns),
    )

    adata.layers["C"] = C
    adata.layers["T"] = T
    adata.layers["total"] = total

    recompute_ntr_and_qc(adata)
    return adata


def recompute_ntr_and_qc(adata: sc.AnnData) -> None:
    total = adata.layers["total"]
    C = adata.layers["C"]

    ntr = np.where(total != 0, C / total, 0)
    adata.layers["ntr"] = ntr
    adata.obs["mean_ntr"] = ntr.mean(axis=1)

    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )


def apply_qc_filters(adata: sc.AnnData) -> sc.AnnData:
    adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs["total_counts"] < 5000, :]
    adata = adata[adata.obs["n_genes_by_counts"] < 2500, :]
    adata = adata[adata.obs["pct_counts_mt"] < 5, :]

    recompute_ntr_and_qc(adata)
    return adata


def export_full_matrices(adata_qc_raw: sc.AnnData, out) -> None:
    C_full = adata_qc_raw.layers["C"]
    T_full = adata_qc_raw.layers["T"]

    C_full = C_full.toarray() if sparse.issparse(C_full) else C_full
    T_full = T_full.toarray() if sparse.issparse(T_full) else T_full

    newrna_full_df = pd.DataFrame(
        C_full.T,
        index=adata_qc_raw.var_names,
        columns=adata_qc_raw.obs_names,
    )
    oldrna_full_df = pd.DataFrame(
        T_full.T,
        index=adata_qc_raw.var_names,
        columns=adata_qc_raw.obs_names,
    )

    out.to_csv_named(newrna_full_df, "newrna_full.csv")
    out.to_csv_named(oldrna_full_df, "oldrna_full.csv")
    print("Exported FULL matrices: newrna_full.csv and oldrna_full.csv")


def run_umap_pipeline(adata_qc_filtered: sc.AnnData, out) -> sc.AnnData:
    adata_umap = adata_qc_filtered.copy()

    sc.pp.normalize_total(adata_umap, target_sum=1e4)
    sc.pp.log1p(adata_umap)

    adata_umap.raw = adata_umap.copy()

    sc.pp.highly_variable_genes(adata_umap, flavor="seurat_v3", n_top_genes=2000)
    adata_umap = adata_umap[:, adata_umap.var.highly_variable]

    sc.pp.scale(adata_umap, max_value=10)
    sc.tl.pca(adata_umap, n_comps=50)

    sc.pl.pca_variance_ratio(adata_umap, log=True, show=False)
    import matplotlib.pyplot as plt

    plt.gca().set_ylabel("Explained variance ratio (log10)")
    plt.tight_layout()
    out.savefig_named("pcas.svg", format="svg")
    plt.show()

    sc.pp.neighbors(adata_umap, n_neighbors=30, n_pcs=30)

    try:
        sc.tl.umap(adata_umap, min_dist=0.05, spread=1.2, random_state=42)
    except TypeError:
        sc.tl.umap(adata_umap)

    sc.tl.leiden(adata_umap, resolution=2)
    return adata_umap


def compute_global_umap_limits(X_umap):
    x = X_umap[:, 0]
    y = X_umap[:, 1]

    x_lo, x_hi = np.percentile(x, [0.2, 99.8])
    y_lo, y_hi = np.percentile(y, [0.2, 99.8])

    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    side = max(x_hi - x_lo, y_hi - y_lo) * 1.18

    xlim_glob = (cx - side / 2, cx + side / 2)
    ylim_glob = (cy - side / 2, cy + side / 2)
    return xlim_glob, ylim_glob


def annotate_cell_states(adata_umap: sc.AnnData) -> sc.AnnData:
    available_2c = [g for g in TWOC_MARKERS if g in adata_umap.raw.var_names]
    available_pluri = [g for g in PLURI_MARKERS if g in adata_umap.raw.var_names]

    print(f"Using 2C markers: {available_2c}")
    print(f"Using pluripotent markers: {available_pluri}")

    if len(available_2c) > 0:
        _2c_vals = np.asarray(adata_umap.raw[:, available_2c].X.mean(axis=1)).ravel()
    else:
        _2c_vals = np.zeros(adata_umap.n_obs)

    if len(available_pluri) > 0:
        _pluri_vals = np.asarray(adata_umap.raw[:, available_pluri].X.mean(axis=1)).ravel()
    else:
        _pluri_vals = np.zeros(adata_umap.n_obs)

    adata_umap.obs["2C_score"] = _2c_vals
    adata_umap.obs["Pluri_score"] = _pluri_vals
    adata_umap.obs["transition_index"] = adata_umap.obs["2C_score"] - adata_umap.obs["Pluri_score"]

    best_score = np.inf
    best_cuts = (0.0, 0.0)
    ti = adata_umap.obs["transition_index"].values

    for inter_cut in np.linspace(-1, 1, 200):
        for c2c_cut in np.linspace(inter_cut + 0.1, 2, 200):
            cats = np.full(len(adata_umap), "Pluripotent", dtype=object)
            cats[(ti > inter_cut) & (ti <= c2c_cut)] = "Intermediate"
            cats[ti > c2c_cut] = "2-cell like"

            vals, counts = np.unique(cats, return_counts=True)
            props = np.zeros(3)

            for i, label in enumerate(STATE_CATEGORIES):
                if label in vals:
                    props[i] = 100 * counts[vals.tolist().index(label)] / len(ti)

            mse = np.mean((props - TARGET_PROPORTIONS) ** 2)
            if mse < best_score:
                best_score = mse
                best_cuts = (inter_cut, c2c_cut)

    print(
        f"\nBest match with article at:\n"
        f"inter_cutoff = {best_cuts[0]:.3f}, 2C_cutoff = {best_cuts[1]:.3f}"
    )

    adata_umap.obs["cell_state"] = "Pluripotent"
    adata_umap.obs.loc[adata_umap.obs["transition_index"] > C2C_CUTOFF, "cell_state"] = "2-cell like"
    adata_umap.obs.loc[
        (adata_umap.obs["transition_index"] > INTER_CUTOFF)
        & (adata_umap.obs["transition_index"] <= C2C_CUTOFF),
        "cell_state",
    ] = "Intermediate"

    adata_umap.obs["cell_state"] = pd.Categorical(
        adata_umap.obs["cell_state"],
        categories=STATE_CATEGORIES,
        ordered=True,
    )
    return adata_umap


def attach_cell_states_to_raw(adata_qc_raw: sc.AnnData, adata_umap: sc.AnnData) -> None:
    adata_qc_raw.obs["cell_state"] = adata_umap.obs["cell_state"].reindex(adata_qc_raw.obs_names)


def compute_gene_kinetics(adata_qc_raw: sc.AnnData, t_label: int = T_LABEL_HOURS) -> None:
    new_raw = np.asarray(adata_qc_raw.layers["C"].sum(axis=0), dtype=float).flatten()
    raw_total = np.asarray(
        (adata_qc_raw.layers["C"] + adata_qc_raw.layers["T"]).sum(axis=0),
        dtype=float,
    ).flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(new_raw, raw_total, out=np.zeros_like(new_raw), where=raw_total > 0)

    ratio = np.clip(ratio, 0.0, 1.0 - 1e-12)
    valid = (raw_total > 0) & (ratio > 0) & (ratio < 1.0 - 1e-12)

    half_life = np.full_like(raw_total, np.nan, dtype=float)
    half_life[valid] = -t_label * np.log(2) / np.log1p(-ratio[valid])

    deg_rate = np.full_like(half_life, np.nan, dtype=float)
    deg_rate[valid] = np.log(2) / half_life[valid]

    new_mean = np.asarray(adata_qc_raw.layers["C"].mean(axis=0), dtype=float).flatten()
    exp_term = np.exp(-deg_rate * t_label)

    synth_rate = np.full_like(deg_rate, np.nan, dtype=float)
    ok = np.isfinite(deg_rate) & (exp_term != 1)
    synth_rate[ok] = (new_mean[ok] * deg_rate[ok]) / (1 - exp_term[ok])

    adata_qc_raw.var["half_life_hr"] = half_life
    adata_qc_raw.var["deg_rate"] = deg_rate
    adata_qc_raw.var["synth_rate"] = synth_rate

    print(adata_qc_raw.var["half_life_hr"].head())


def export_kinetics_tables(adata_qc_raw: sc.AnnData, out) -> None:
    kinetics = adata_qc_raw.var[["half_life_hr", "deg_rate", "synth_rate"]].copy()
    kinetics.index.name = "Gene"
    kinetics = kinetics.replace([np.inf, -np.inf], np.nan)

    out.to_csv_named(kinetics, "gene_kinetics_all.csv")
    kinetics_valid = kinetics.dropna().query("half_life_hr > 0 and half_life_hr < 24")
    out.to_csv_named(kinetics_valid, "gene_kinetics_valid_0-24h.csv")


def build_half_life_merge(adata_qc_raw: sc.AnnData, slam_df: pd.DataFrame, hl_max=24) -> pd.DataFrame:
    our = adata_qc_raw.var[["half_life_hr"]].reset_index().rename(columns={"index": "Gene"})
    merged = pd.merge(slam_df, our, on="Gene", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["SLAM_half_life_hr", "half_life_hr"]
    )
    merged = merged[
        (merged["SLAM_half_life_hr"] > 0)
        & (merged["SLAM_half_life_hr"] <= hl_max)
        & (merged["half_life_hr"] > 0)
        & (merged["half_life_hr"] <= hl_max)
    ]

    print(f"[half-life corr] n={len(merged)} genes after filtering to 0–{hl_max} h")
    return merged


def half_life_from_mask(adata_qc_raw: sc.AnnData, mask_bool, t_label: int = T_LABEL_HOURS):
    C = adata_qc_raw.layers["C"]
    T = adata_qc_raw.layers["T"]

    if sparse.issparse(C):
        C = C.toarray()
    if sparse.issparse(T):
        T = T.toarray()

    Csum = C[mask_bool, :].sum(axis=0).astype(float)
    Tsum = T[mask_bool, :].sum(axis=0).astype(float)
    Tot = Csum + Tsum

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(Csum, Tot, out=np.zeros_like(Csum), where=Tot > 0)

    ratio = np.clip(ratio, 0.0, 1.0 - 1e-12)
    hl = np.full_like(Tot, np.nan, dtype=float)
    ok = (Tot > 0) & (ratio > 0) & (ratio < 1.0 - 1e-12)
    hl[ok] = -t_label * np.log(2) / np.log1p(-ratio[ok])
    return hl


def subsample_stability_holdout(
    adata_qc_raw: sc.AnnData,
    state: str,
    k_grid,
    B=50,
    seed=42,
):
    rng = np.random.default_rng(seed)
    idx_state = np.where(adata_qc_raw.obs["cell_state"].values == state)[0]
    n = idx_state.size

    if n < 3:
        print(f"[skip] {state}: too few cells ({n}).")
        return {}

    ks = [k for k in k_grid if k <= n - 2]
    results = {k: [] for k in ks}

    for k in ks:
        for _ in range(B):
            take = rng.choice(idx_state, size=k, replace=False)

            ref_mask = np.zeros(adata_qc_raw.n_obs, dtype=bool)
            ref_mask[idx_state] = True
            ref_mask[take] = False

            sub_mask = np.zeros_like(ref_mask)
            sub_mask[take] = True

            hl_ref = half_life_from_mask(adata_qc_raw, ref_mask)
            hl_sub = half_life_from_mask(adata_qc_raw, sub_mask)

            ok = np.isfinite(hl_ref) & np.isfinite(hl_sub)
            if ok.sum() >= 50:
                r, _ = pearsonr(hl_ref[ok], hl_sub[ok])
                results[k].append(r)

    return results


def merge_qiu_rates(qiu_df: pd.DataFrame, adata_qc_raw: sc.AnnData):
    my_deg = (
        adata_qc_raw.var[["deg_rate"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index()
        .rename(columns={"index": "Gene", "deg_rate": "Our_deg_rate"})
    )
    merged_deg = pd.merge(qiu_df, my_deg, on="Gene").rename(
        columns={"Degradation_rate_Pluripotent": "scNTseq_deg_rate"}
    )
    merged_deg = merged_deg[(merged_deg["scNTseq_deg_rate"] <= 1)]

    my_synth = (
        adata_qc_raw.var[["synth_rate"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index()
        .rename(columns={"index": "Gene", "synth_rate": "Our_synth_rate"})
    )
    merged_synth = pd.merge(qiu_df, my_synth, on="Gene").rename(
        columns={"Synthesis_rate_Pluripotent": "scNTseq_synth_rate"}
    )

    return merged_deg, merged_synth


def export_per_state_full_allgenes(adata_qc_raw: sc.AnnData, out) -> None:
    adata_qc_raw.var["gene_symbol"] = adata_qc_raw.var_names

    out.write_h5ad_named(
        adata_qc_raw,
        "adata_qc_raw_with_kinetics_and_states.h5ad",
        compression="gzip",
    )

    for layer_name in ["C", "T", "total", "ntr"]:
        if layer_name in adata_qc_raw.layers and adata_qc_raw.layers[layer_name].dtype != np.float32:
            adata_qc_raw.layers[layer_name] = adata_qc_raw.layers[layer_name].astype(np.float32)

    for state in adata_qc_raw.obs["cell_state"].cat.categories:
        mask = (adata_qc_raw.obs["cell_state"] == state).values
        adata_state = adata_qc_raw[mask, :].copy()

        if "total" not in adata_state.layers:
            adata_state.layers["total"] = adata_state.layers["C"] + adata_state.layers["T"]
        if "ntr" not in adata_state.layers:
            denom = np.clip(adata_state.layers["total"], 1e-12, None)
            adata_state.layers["ntr"] = adata_state.layers["C"] / denom

        tag = STATE_SLUG[state]
        out.write_h5ad_named(adata_state, f"adata_{tag}_full_allgenes.h5ad", compression="gzip")
        print(f"Wrote per-state AnnData (ALL genes): adata_{tag}_full_allgenes.h5ad")

        M_total = adata_state.layers["total"]
        M_ntr = adata_state.layers["ntr"]
        if sparse.issparse(M_total):
            M_total = M_total.toarray()
        if sparse.issparse(M_ntr):
            M_ntr = M_ntr.toarray()

        total_df = pd.DataFrame(M_total.T, index=adata_state.var_names, columns=adata_state.obs_names)
        ntr_df = pd.DataFrame(M_ntr.T, index=adata_state.var_names, columns=adata_state.obs_names)

        out.to_csv_named(total_df, f"totalrna_{tag}_full_allgenes.csv")
        out.to_csv_named(ntr_df, f"ntr_{tag}_full_allgenes.csv")
        print(f"Exported {state}: totalrna_{tag}_full_allgenes.csv and ntr_{tag}_full_allgenes.csv")


def build_per_state_dropout_df(
    adata_qc_raw: sc.AnnData,
    df_no4su: pd.DataFrame,
    min_det=0.02,
    winsor=6.0,
):
    C = adata_qc_raw.layers["C"]
    T = adata_qc_raw.layers["T"]

    if sparse.issparse(C):
        C = C.toarray()
    if sparse.issparse(T):
        T = T.toarray()

    states = list(adata_qc_raw.obs["cell_state"].cat.categories)

    shared = adata_qc_raw.var_names.intersection(df_no4su.columns)
    idx = adata_qc_raw.var_names.get_indexer(shared)

    C_all = C[:, idx]
    T_all = T[:, idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        ntr_global_vec = np.where(
            (C_all.sum(axis=0) + T_all.sum(axis=0)) > 0,
            C_all.sum(axis=0) / (C_all.sum(axis=0) + T_all.sum(axis=0)),
            np.nan,
        )
    ntr_global = pd.Series(np.asarray(ntr_global_vec).flatten(), index=shared).dropna()

    shared = ntr_global.index
    idx = adata_qc_raw.var_names.get_indexer(shared)
    ntr_rank_global = ntr_global.rank(ascending=False, method="average")

    X0 = df_no4su[shared].to_numpy()
    lib0 = X0.sum(axis=1, keepdims=True)
    frac0 = (X0 / np.clip(lib0, 1, None)).mean(axis=0)
    det0 = (X0 > 0).mean(axis=0)

    eps = 1e-12
    frac0 = np.clip(frac0, eps, None)

    frames = []
    stats_rows = []

    for state in states:
        mask = (adata_qc_raw.obs["cell_state"] == state).values
        Xs = (C[mask, :][:, idx] + T[mask, :][:, idx])
        libs = Xs.sum(axis=1, keepdims=True)
        fracs = Xs / np.clip(libs, 1, None)
        frac_s = fracs.mean(axis=0)
        det_s = (Xs > 0).mean(axis=0)

        frac_s = np.clip(np.asarray(frac_s).flatten(), eps, None)
        det_s = np.asarray(det_s).flatten()

        keep = (det0 >= min_det) & (det_s >= min_det)
        g = np.array(shared)[keep]
        f0 = frac0[keep]
        fs = frac_s[keep]

        log2fc = np.log2(fs / f0)
        log2fc_centered = log2fc - np.median(log2fc)
        if winsor is not None:
            log2fc_centered = np.clip(log2fc_centered, -winsor, winsor)

        df = pd.DataFrame(
            {
                "gene": g,
                "ntr_rank_order": ntr_rank_global.loc[g].values,
                "log2FC_centered": log2fc_centered,
                "state": state,
            }
        )
        frames.append(df)

        rho, p = spearmanr(df["ntr_rank_order"], df["log2FC_centered"])
        stats_rows.append((state, df.shape[0], float(np.median(log2fc_centered)), float(rho), float(p)))

    df_state_diag = pd.concat(frames, ignore_index=True)
    stats_df = pd.DataFrame(
        stats_rows,
        columns=["state", "n_genes", "median_log2FC_centered", "spearman_rho", "p_value"],
    )
    return df_state_diag, stats_df


def compute_kinetics_by_state(
    adata_qc_raw: sc.AnnData,
    t_label: int = T_LABEL_HOURS,
) -> pd.DataFrame:
    C_ks = adata_qc_raw.layers["C"]
    T_ks = adata_qc_raw.layers["T"]

    if sparse.issparse(C_ks):
        C_ks = C_ks.toarray()
    if sparse.issparse(T_ks):
        T_ks = T_ks.toarray()

    genes = adata_qc_raw.var_names.to_numpy()
    states = list(adata_qc_raw.obs["cell_state"].cat.categories)

    rows = []
    for state in states:
        mask = (adata_qc_raw.obs["cell_state"] == state).values

        Csum = C_ks[mask, :].sum(axis=0).astype(float)
        Tsum = T_ks[mask, :].sum(axis=0).astype(float)
        Tot = Csum + Tsum

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(Csum, Tot, out=np.zeros_like(Csum), where=Tot > 0)

        ratio = np.clip(ratio, 0.0, 1.0 - 1e-12)
        valid = (Tot > 0) & (ratio > 0) & (ratio < 1.0 - 1e-12)

        hl = np.full_like(Tot, np.nan, dtype=float)
        hl[valid] = -t_label * np.log(2) / np.log1p(-ratio[valid])

        deg = np.full_like(hl, np.nan, dtype=float)
        deg[valid] = np.log(2) / hl[valid]

        new_mean_state = C_ks[mask, :].mean(axis=0).astype(float)
        exp_term = np.exp(-deg * t_label)
        synth = np.full_like(deg, np.nan, dtype=float)
        ok = np.isfinite(deg) & (exp_term != 1)
        synth[ok] = (new_mean_state[ok] * deg[ok]) / (1 - exp_term[ok])

        rows.append(
            pd.DataFrame(
                {
                    "gene": genes,
                    "state": state,
                    "half_life_hr": hl,
                    "deg_rate": deg,
                    "synth_rate": synth,
                    "Csum": Csum,
                    "Tsum": Tsum,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def export_hvg_inputs(adata_qc_raw: sc.AnnData, adata_umap: sc.AnnData, out) -> None:
    hvg_gene_names = adata_umap.var_names[adata_umap.var["highly_variable"]]
    adata_allhvg = adata_qc_raw[adata_umap.obs_names, hvg_gene_names].copy()

    adata_allhvg.obs["cell_state"] = adata_umap.obs["cell_state"].reindex(adata_allhvg.obs_names)

    C = adata_allhvg.layers["C"]
    T = adata_allhvg.layers["T"]
    C_dense = C.toarray() if sparse.issparse(C) else C
    T_dense = T.toarray() if sparse.issparse(T) else T

    pd.DataFrame(C_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names).to_csv(
        out.out_path(out.add_suffix("newrna_hvg.csv"))
    )
    pd.DataFrame(T_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names).to_csv(
        out.out_path(out.add_suffix("oldrna_hvg.csv"))
    )
    print("Exported HVG-based NASC-seq2 input: newrna_hvg.csv and oldrna_hvg.csv")

    for state in adata_allhvg.obs["cell_state"].cat.categories:
        mask = adata_allhvg.obs["cell_state"] == state
        selected_cells = adata_allhvg.obs_names[mask]

        C_sub = C_dense[mask, :]
        T_sub = T_dense[mask, :]

        C_df = pd.DataFrame(C_sub.T, index=adata_allhvg.var_names, columns=selected_cells)
        T_df = pd.DataFrame(T_sub.T, index=adata_allhvg.var_names, columns=selected_cells)

        tag = STATE_SLUG[state]
        C_df.to_csv(out.out_path(out.add_suffix(f"newrna_{tag}_hvg.csv")))
        T_df.to_csv(out.out_path(out.add_suffix(f"oldrna_{tag}_hvg.csv")))
        print(f"Exported {state}: newrna_{tag}_hvg.csv and oldrna_{tag}_hvg.csv")

    C_full = adata_qc_raw.layers["C"]
    T_full = adata_qc_raw.layers["T"]
    C_full_dense = C_full.toarray() if sparse.issparse(C_full) else C_full
    T_full_dense = T_full.toarray() if sparse.issparse(T_full) else T_full

    for state in adata_umap.obs["cell_state"].cat.categories:
        mask = adata_umap.obs["cell_state"] == state
        selected_cells = adata_umap.obs_names[mask]
        selected_cells = selected_cells.intersection(adata_qc_raw.obs_names)
        idx_rows = adata_qc_raw.obs_names.get_indexer(selected_cells)

        C_sub = C_full_dense[idx_rows, :]
        T_sub = T_full_dense[idx_rows, :]

        C_df_full = pd.DataFrame(C_sub.T, index=adata_qc_raw.var_names, columns=selected_cells)
        T_df_full = pd.DataFrame(T_sub.T, index=adata_qc_raw.var_names, columns=selected_cells)

        tag = STATE_SLUG[state]
        C_df_full.to_csv(out.out_path(out.add_suffix(f"newrna_{tag}_full.csv")))
        T_df_full.to_csv(out.out_path(out.add_suffix(f"oldrna_{tag}_full.csv")))
        print(f"Exported {state}: newrna_{tag}_full.csv and oldrna_{tag}_full.csv")

    all_C_df = pd.DataFrame(C_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names)
    all_T_df = pd.DataFrame(T_dense.T, index=adata_allhvg.var_names, columns=adata_allhvg.obs_names)

    all_C_df.to_csv(out.out_path(out.add_suffix("newrna_allcells_hvg.csv")))
    all_T_df.to_csv(out.out_path(out.add_suffix("oldrna_allcells_hvg.csv")))
    print("Exported NASC-seq2 fitting input for all cells: newrna_allcells_hvg.csv and oldrna_allcells_hvg.csv")