from pathlib import Path

from config import (
    MARKERS,
    ProjectPaths,
    STATE_CATEGORIES,
    apply_plot_styles,
)
from io_utils import (
    OutputManager,
    load_count_tables,
    load_no4su_counts,
    load_qiu_rate_reference,
    load_slam_reference,
)
from pipeline import (
    annotate_cell_states,
    apply_qc_filters,
    attach_cell_states_to_raw,
    build_half_life_merge,
    build_initial_adata,
    build_per_state_dropout_df,
    compute_gene_kinetics,
    compute_global_umap_limits,
    compute_kinetics_by_state,
    export_full_matrices,
    export_hvg_inputs,
    export_kinetics_tables,
    export_per_state_full_allgenes,
    merge_qiu_rates,
    run_umap_pipeline,
    subsample_stability_holdout,
)
from plotting import (
    plot_cell_state_umap,
    plot_dropout_vs_ntr_rank,
    plot_half_life_vs_slam,
    plot_marker_umaps,
    plot_qc_scatter,
    plot_qc_violins,
    plot_rate_comparison,
    plot_scalar_umaps,
    plot_stability_results,
    plot_transition_index_umap,
)


def main():
    apply_plot_styles()

    project_root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths(project_root=project_root)
    out = OutputManager(paths.results_dir, paths.suffix)

    print(f"All outputs will be written to: {paths.results_dir} with suffix {paths.suffix}")

    # ==============================
    # Step 1: Load data
    # ==============================
    c_df, t_df = load_count_tables(paths)
    adata_initial = build_initial_adata(c_df, t_df)
    adata_pre = adata_initial.copy()

    # ==============================
    # Step 2: Quality control filtering
    # ==============================
    adata_filtered = apply_qc_filters(adata_initial)

    plot_qc_violins(
        adata_filtered,
        out,
        filename="violin_plots_POSTQC_cutoffs.svg",
        pre_or_post_label="POST",
    )
    plot_qc_violins(
        adata_pre,
        out,
        filename="violin_plots_PREQC_cutoffs.svg",
        pre_or_post_label="PRE",
    )
    plot_qc_scatter(adata_filtered, out, filename="scatter_plots.svg")

    # ==============================
    # Step 3: Split working objects
    # ==============================
    adata_qc_raw = adata_filtered.copy()
    adata_umap = adata_filtered.copy()

    export_full_matrices(adata_qc_raw, out)

    # ==============================
    # Step 4: UMAP + Leiden + manual state annotation
    # ==============================
    adata_umap = run_umap_pipeline(adata_umap, out)
    X_umap = adata_umap.obsm["X_umap"]
    xlim_glob, ylim_glob = compute_global_umap_limits(X_umap)

    plot_scalar_umaps(adata_umap, X_umap, xlim_glob, ylim_glob, out)
    plot_marker_umaps(adata_umap, X_umap, xlim_glob, ylim_glob, MARKERS, out)

    adata_umap = annotate_cell_states(adata_umap)

    plot_transition_index_umap(adata_umap, X_umap, xlim_glob, ylim_glob, out)
    plot_cell_state_umap(adata_umap, X_umap, xlim_glob, ylim_glob, out)

    out.write_h5ad_named(adata_umap, "adata_umap_with_states.h5ad", compression="gzip")

    print((adata_umap.obs["cell_state"].value_counts(normalize=True) * 100).round(2))
    counts = adata_umap.obs["cell_state"].value_counts()
    percentages = (counts / counts.sum()) * 100
    print(percentages)

    # ==============================
    # Step 5: Gene-level kinetics
    # ==============================
    attach_cell_states_to_raw(adata_qc_raw, adata_umap)
    compute_gene_kinetics(adata_qc_raw)
    export_kinetics_tables(adata_qc_raw, out)

    slam_df = load_slam_reference(paths)
    half_life_merge = build_half_life_merge(adata_qc_raw, slam_df, hl_max=24)
    plot_half_life_vs_slam(half_life_merge, out, hl_max=24)

    # ==============================
    # Step 6: Subsampling stability
    # ==============================
    res_pluri = subsample_stability_holdout(
        adata_qc_raw,
        "Pluripotent",
        k_grid=(5, 10, 20, 30, 40, 60, 80, 120, 200, 400),
        B=30,
    )
    res_2c = subsample_stability_holdout(
        adata_qc_raw,
        "2-cell like",
        k_grid=(3, 5, 8, 10, 12),
        B=200,
    )
    plot_stability_results(res_pluri, res_2c, out)

    # ==============================
    # Step 7: Deg / synth rate comparisons
    # ==============================
    qiu_df = load_qiu_rate_reference(paths)
    merged_deg, merged_synth = merge_qiu_rates(qiu_df, adata_qc_raw)

    plot_rate_comparison(
        merged_deg,
        x_col="scNTseq_deg_rate",
        y_col="Our_deg_rate",
        title="Degradation Rate Comparison",
        xlabel="scNTseq degradation rate (hr⁻¹)",
        ylabel="Estimated degradation rate (hr⁻¹)",
        out_filename="degradation_rate_comparison.svg",
        out=out,
    )

    plot_rate_comparison(
        merged_synth,
        x_col="scNTseq_synth_rate",
        y_col="Our_synth_rate",
        title="Synthesis Rate Comparison",
        xlabel="scNTseq synthesis rate (hr⁻¹)",
        ylabel="Estimated synthesis rate (hr⁻¹)",
        out_filename="synthesis_rate_comparison.svg",
        out=out,
    )

    # ==============================
    # Step 8: Per-state exports
    # ==============================
    export_per_state_full_allgenes(adata_qc_raw, out)

    # ==============================
    # Step 9: Dropout diagnostics
    # ==============================
    df_no4su = load_no4su_counts(paths)
    df_state_diag, stats_df = build_per_state_dropout_df(
        adata_qc_raw,
        df_no4su,
        min_det=0.02,
        winsor=6.0,
    )
    print(stats_df)
    plot_dropout_vs_ntr_rank(df_state_diag, STATE_CATEGORIES, out)

    # ==============================
    # Step 10: Per-state kinetic parameter exports
    # ==============================
    kin_by_state = compute_kinetics_by_state(adata_qc_raw)
    out.to_csv_named(kin_by_state, "gene_kinetics_by_state.csv")

    # ==============================
    # Step 11: HVG exports for NASC-seq2 inputs
    # ==============================
    export_hvg_inputs(adata_qc_raw, adata_umap, out)

    print("Done. Same science, less script sludge.")


if __name__ == "__main__":
    main()