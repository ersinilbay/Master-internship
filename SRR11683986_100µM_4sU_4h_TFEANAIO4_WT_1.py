import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/Users/ersinilbay/Downloads/mESC-WT-rep1_C.txt"
adata = sc.read_text(file_path)

adata.raw = adata  # Store raw data before modifications

# adata.layers["ntr"] = ntr_matrix # add NTR layer to Anndata
# raw data layer
# new rnas layer

adata = adata.T  # Swap rows and columns because the data is flipped
print(adata)
print(adata.obs_names)
print("Shape of adata:", adata.shape)
print("First 5 cell barcodes (obs_names):", adata.obs_names[:5])  # Should be cell barcodes
print("First 5 gene names (var_names):", adata.var_names[:5])  # Should be gene names

# Identify mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("mt-")

# Calculate QC metrics, including % mitochondrial content
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# # Filter out cells with few genes and genes detected in too few cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
#
# # Filter out cells with high counts
adata = adata[adata.obs['total_counts'] < 5000, :]
adata = adata[adata.obs['n_genes_by_counts'] < 2500, :]
#
# #Filter out cells with high % mito genes
adata = adata[adata.obs['pct_counts_mt'] < 5, :]

print(adata.X[:5, :5])  # Look at first few values
print(adata.obs.shape[0]) # amount of cells in the dataset

# violin plots
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, log=False, multi_panel=True)
# sns.stripplot(data=df, jitter=True, size=1, alpha=0.5, color="black")
# plt.show()

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot 1: Total counts vs number of genes
sc.pl.scatter(
    adata, x='total_counts', y='n_genes_by_counts', ax=axes[0],
    title="Total Counts vs Number of Genes", size=50, show=False
)
plt.show()
# Scatter plot 2: Total counts vs % mitochondrial genes
sc.pl.scatter(
    adata, x='total_counts', y='pct_counts_mt', ax=axes[1],
    title="Total Counts vs % Mitochondrial Genes", size=50, show=False
)

# Adjust grid
for ax in axes:
    ax.grid(alpha=0.3)  # Make the grid lighter

plt.show()

sc.pp.normalize_total(adata, target_sum=1e4) #total count normalize the data matrix to 10000 reads per cell

sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

sc.pl.highly_variable_genes(adata)
