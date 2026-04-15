# Refactored QC Workflow

This folder contains a refactored version of the QC, state-annotation, and RNA-kinetics workflow developed during my MSc internship on scNT-seq data from mouse embryonic stem cells (mESCs).

The workflow is not just used to generate figures. It serves as the bridge between raw scNT-seq input and the biological interpretation in the internship project: starting from paired new and old RNA count matrices, it performs QC, cell-state annotation, RNA-stability validation, and exports processed data for downstream variability and burst-kinetics analyses.

## Biological objective

The biological goal of this workflow is to process 4sU/scNT-seq data from mESCs and generate a structured, interpretable representation of the dataset that can support questions about:

- cell-state heterogeneity
- transitions toward a 2-cell-like state
- RNA stability and kinetic behavior
- downstream transcriptional variability analyses

In practice, this workflow transforms raw paired labeled/unlabeled count matrices into:

- quality-controlled single-cell objects
- state-annotated cell populations
- gene-level kinetic estimates
- processed exports for downstream analysis

## My contribution

During my internship, I implemented the computational workflow in Python to connect raw scNT-seq count matrices to biologically interpretable outputs.

This included:

- constructing an `AnnData` object from paired new (`C`) and old (`T`) RNA count matrices
- computing quality-control metrics
- filtering low-quality cells and low-information genes
- performing PCA / UMAP / Leiden-based structure analysis
- annotating cells into `Pluripotent`, `Intermediate`, and `2-cell like` states using marker-based scores
- estimating gene-level half-life, degradation rate, and synthesis rate
- comparing inferred kinetic quantities to external reference datasets
- exporting processed matrices and `.h5ad` objects for downstream analyses

The code in this folder is a post-internship refactor of the original working script, created to improve readability and reproducibility while keeping the intended analysis logic and outputs as close as possible to the original workflow.

## Workflow overview

### Inputs
The workflow starts from paired gene-by-cell UMI matrices:

- `C`: newly labeled / new RNA counts
- `T`: pre-existing / old RNA counts

These matrices are used to build a layered `AnnData` object containing:

- `C`
- `T`
- `total`
- `ntr`

### Core analysis steps

1. **Build the base single-cell object**
   - construct `AnnData` from paired new/old RNA matrices

2. **Compute QC metrics**
   - genes detected per cell
   - total UMI counts
   - mitochondrial fraction
   - mean NTR per cell

3. **Apply QC filtering**
   - filter low-quality cells
   - filter low-information genes

4. **Run dimensionality reduction and clustering**
   - normalization
   - log transformation
   - HVG selection
   - PCA
   - neighborhood graph
   - UMAP
   - Leiden clustering

5. **Annotate cell states**
   - score cells using pluripotency and 2-cell-like marker sets
   - define `Pluripotent`, `Intermediate`, and `2-cell like` states

6. **Estimate RNA kinetic quantities**
   - half-life
   - degradation rate
   - synthesis rate

7. **Validate against external references**
   - compare estimated half-lives and rates to published reference datasets

8. **Export processed outputs**
   - figures
   - kinetic summary tables
   - state-annotated `.h5ad` files
   - matrix exports for downstream variability / burst-kinetics analyses

## Repository structure

### `run_qc_report.py`
Main entry point for the workflow.  
Runs the full pipeline from loading data to export of figures and processed outputs.

### `config.py`
Central configuration file.  
Defines paths, filenames, marker sets, plotting settings, state labels, and analysis constants.

### `io_utils.py`
Input/output helper functions.  
Loads required files and saves figures, tables, and `.h5ad` outputs with consistent naming.

### `plotting.py`
Plotting helper functions.  
Contains reusable routines for QC figures, UMAP visualizations, validation plots, and diagnostic plots.

### `pipeline.py`
Core analysis logic.  
Contains the main computational steps for QC, dimensionality reduction, annotation, kinetic estimation, validation, and export.

## Input files

The workflow expects a `data/` folder at the repository root containing the required input files.

Expected filenames:

- `mESC-WT-rep1_C.txt`
- `mESC-WT-rep1_T.txt`
- `41592_2017_BFnmeth4435_MOESM4_ESM.xls`
- `scNTseq_params.xlsx`
- `GSM4671630_CK-TFEA-run1n2_ds3_gene_exonic.intronic_tagged.dge.txt`

These files are not included in the public repository.

### Provenance
The local input/reference files used here are based on internship data and external reference material derived from published studies, including:

- scNT-seq data from Qiu et al.
- SLAM-seq reference material used for half-life comparison
- additional external processed reference files used in stability/dropout validation

## Main outputs

The workflow writes results to the configured results directory, for example:

`results/_rep1_fix/`

Outputs include:

- QC violin and scatter plots
- PCA variance plot
- UMAP visualizations
- cell-state annotation outputs
- half-life / degradation / synthesis summary tables
- state-annotated `.h5ad` objects
- per-state matrix exports
- HVG-based exports for downstream analyses

## Example figures

Example visualizations can be stored in `examples/` and embedded here.

Suggested examples:
- a QC plot
- a cell-state UMAP
- a half-life validation plot

For example:

```md
![QC violin example](examples/violin_plots_POSTQC_cutoffs.png)
![Cell-state UMAP example](examples/umap_cell_states.png)
![Half-life validation example](examples/half_life_vs_SLAM_0-24h.png)