#!/usr/bin/env bash
# NASC-seq2 end-to-end with two lookup grids.
# HIGH grid is used for primary inference (more stable/accurate); MED grid is an optional sensitivity check.
# ML is seeded from HIGH lookup results.

set -euo pipefail
conda activate nascseq2 >/dev/null 2>&1 || true

# ============= 0) INPUTS — EDIT THESE =====================================
NEWRNA="/abs/path/to/newrna.csv"  # per-subset new RNA counts (genes × cells)
OLDRNA="/abs/path/to/oldrna.csv"  # per-subset old RNA counts (genes × cells)
TIME_H=4                          # labeling/incubation time (hours) for this subset
BASE="/mbshome/eilbay/NASC-seq2/RUN_EXAMPLE"   # run folder
mkdir -p "$BASE"
export NEWRNA OLDRNA TIME_H BASE

# Use a single thread for math libraries to keep results reproducible and avoid CPU overuse across all steps.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============= 1) ESTIMATE DEGRADATION RATE D (subset-specific) ============
# This computes D for the current subset; downstream steps read it from $D.
python3 - <<'PY' | tee "$BASE/degradation_rate.txt"
import os, pandas as pd, numpy as np, math
NEWRNA, OLDRNA = os.environ["NEWRNA"], os.environ["OLDRNA"]
T = float(os.environ["TIME_H"])

# Load and align gene sets.
n = pd.read_csv(NEWRNA, index_col=0)
o = pd.read_csv(OLDRNA, index_col=0)
n, o = n.align(o, join="outer", axis=0, fill_value=0)

# Global fraction-new → d_total.
new_tot, old_tot = n.to_numpy().sum(), o.to_numpy().sum()
fnew = new_tot / (new_tot + old_tot) if (new_tot + old_tot) > 0 else float('nan')
d_total = -math.log(max(1e-12, 1.0 - fnew))/T if not math.isnan(fnew) else float('nan')      # used for downstream!

# Per-cell median d (more robust when coverage varies).                                      # sanity check (should be similar to d_total for best results)
f_cell = (n/(n+o).replace(0, np.nan)).mean(axis=0, skipna=True)
d_cells = -np.log((1 - f_cell.clip(0,0.999999))).dropna()/T
d_median = float(d_cells.median()) if len(d_cells) else float('nan')  

# Print summaries and emit the single D we will use on a labeled line.
print(f"fraction_new_total={fnew:.6f}")
print(f"degradation_rate_total={d_total:.12f}")
print(f"degradation_rate_median_per_cell={d_median:.12f}")
print(f"D_used={d_total:.12f}")  # paper/demo choice: global fraction-new
PY
export D="$(grep -Eo 'D_used=([0-9.]+)' "$BASE/degradation_rate.txt" | cut -d= -f2)"
echo ">>> Using D=$D (1/h) and TIME_H=$TIME_H h"

# ============= 2) BUILD LOOKUP GRIDS ======================================
# HIGH: finer grid to get more accurate and stable bootstrap selections. 
# MED: coarser grid used as a quick sensitivity/reproducibility check.

# HIGH grid (denser; 16 cores)
nice -n 10 python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/parameter_table_calc_from_prob_3proxies.py \
  --open_rate        0.001 50 36 \
  --close_rate       0.001 50 22 \
  --transcribe_rate  0.001 1  18 \
  --degrade_rate     "$D" "$D" 1 \
  --time             "$TIME_H" "$TIME_H" 1 \
  --proc 16 \
  --prec 10000 \
  --table_out "$BASE/m_table_HIGH.tsv" |& tee "$BASE/m_table_HIGH.log"

# Trim removes unreachable/duplicate rows to speed up lookups.
python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/trim_lookup_table.py \
  "$BASE/m_table_HIGH.tsv" "$BASE/m_table_HIGH.trimmed.tsv"

# MED grid (coarser; optional; 8 cores)
nice -n 10 python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/parameter_table_calc_from_prob_3proxies.py \
  --open_rate        0.001 50 24 \
  --close_rate       0.001 50 16 \
  --transcribe_rate  0.001 1  12 \
  --degrade_rate     "$D" "$D" 1 \
  --time             "$TIME_H" "$TIME_H" 1 \
  --proc 8 \
  --prec 10000 \
  --table_out "$BASE/m_table_MED.tsv" |& tee "$BASE/m_table_MED.log"

python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/trim_lookup_table.py \
  "$BASE/m_table_MED.tsv" "$BASE/m_table_MED.trimmed.tsv"

# ============= 3) BOOTSTRAP LOOKUP FITS ===================================
# Bootstrap gives per-gene uncertainty and defines “robust” genes before ML.
nice -n 10 python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/bootstrap_nonzero_three_estimate_lookup_one_csv_v5.py \
  "$NEWRNA" \
  "$BASE/kinetics_from_lookup_HIGH.tsv" \
  -m "$BASE/m_table_HIGH.trimmed.tsv" \
  --lookup_mode interp nearest noborders \
  --proc 16 \
  --reuse_lookups \
  --bootstraps 50 |& tee "$BASE/bootstrap_HIGH.log"

# MED (optional sensitivity)
nice -n 10 python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/bootstrap_nonzero_three_estimate_lookup_one_csv_v5.py \
  "$NEWRNA" \
  "$BASE/kinetics_from_lookup_MED.tsv" \
  -m "$BASE/m_table_MED.trimmed.tsv" \
  --lookup_mode interp nearest noborders \
  --proc 8 \
  --reuse_lookups \
  --bootstraps 50 |& tee "$BASE/bootstrap_MED.log"

# ============= 4) ROBUST GENE SELECTION ===================================
# filter robustness: use 50% CIs and log2-width cutoffs from demo (and fits paper IQR rule)

python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/kinetics_plotting/plot_parameter_distributions_with_bootstrap_filters_v3.py \
  "$BASE/kinetics_from_lookup_HIGH.tsv" \
  -o "$BASE/parameter_distributions_HIGH.pdf" \
  --conffilter kon koff ksyn \
  --conf 50 \
  -r 1 2 50 \
  --export_index "$BASE/robust_genes_HIGH.txt"

# Optional sensitivity on MED grid (same demo thresholds)
python3 /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/kinetics_plotting/plot_parameter_distributions_with_bootstrap_filters_v3.py \
  "$BASE/kinetics_from_lookup_MED.tsv" \
  -o "$BASE/parameter_distributions_MED.pdf" \
  --conffilter kon koff ksyn \
  --conf 50 \
  -r 1 2 50 \
  --export_index "$BASE/robust_genes_MED.txt"


# ============= 5) BUILD ML SEEDS (HIGH only) ==============================
python3 - <<'PY'
import os, pandas as pd
B = os.environ["BASE"]
h = pd.read_csv(f"{B}/kinetics_from_lookup_HIGH.tsv", sep='\t', index_col=0)
cols = ["kon","koff","ksyn"]
seeds = h[cols].copy()
# restrict to HIGH robust list (authoritative)
rob = pd.read_csv(f"{B}/robust_genes_HIGH.txt", header=None)[0] if os.path.exists(f"{B}/robust_genes_HIGH.txt") else seeds.index
seeds = seeds.loc[seeds.index.intersection(rob)]
seeds.to_csv(f"{B}/guesses_for_ML.tsv", sep='\t')
print("Wrote", f"{B}/guesses_for_ML.tsv", "rows:", len(seeds))
PY

# ============= 6) SUBSET NEW RNA TO ROBUST GENES (HIGH) ===================
# Limits ML to genes that passed the bootstrap filters on HIGH.
python3 - <<'PY'
import os, pandas as pd
B, NR = os.environ["BASE"], os.environ["NEWRNA"]
genes = pd.read_csv(f"{B}/guesses_for_ML.tsv", sep='\t', index_col=0).index
mat = pd.read_csv(NR, index_col=0)
keep = mat.loc[mat.index.intersection(genes)]
keep.to_csv(f"{B}/newrna_subset_for_ML.csv")
print("Subset:", keep.shape)
PY

# ============= 7) MAXIMUM LIKELIHOOD (continuous fit) =====================
# Refines discrete lookup estimates to continuous optima.
nice -n 10 "$CONDA_PREFIX/bin/python3" /mbshome/eilbay/NASC-seq2/burst_kinetic_parameter_estimation/transient_ml_v2.py \
  -i "$BASE/newrna_subset_for_ML.csv" \
  --guesses "$BASE/guesses_for_ML.tsv" \
  -d "$D" \
  --prec 10000 \
  --time "$TIME_H" \
  --threads 16 \
  -o "$BASE/ML_results.csv" 2>&1 | tee "$BASE/ML_results.log"

# ============= 8) ADD DERIVED METRICS (from ML results) =======================
# Adds common summaries in the same table for downstream use.
python3 - <<'PY'
import os, pandas as pd, numpy as np
B = os.environ["BASE"]
df = pd.read_csv(f"{B}/ML_results.csv", index_col=0)
kon=df["kon"].astype(float); koff=df["koff"].astype(float); ksyn=df["ksyn"].astype(float)
df["burst_size_calc"] = ksyn/koff.replace(0,np.nan)          # burst size = ksyn/koff
df["occupancy"]       = kon/(kon+koff)                        # P(ON)
df["burst_frequency"] = (kon*koff)/(kon+koff)                 # frequency ~ kon*koff/(kon+koff)
df["expression_rate"] = ksyn*df["occupancy"]                  # mean new-RNA rate
df.to_csv(f"{B}/ML_results.with_derived.csv")
print("Wrote", f"{B}/ML_results.with_derived.csv", "rows:", len(df))
PY

# ============= 9) JOIN LOOKUP CIs (from HIGH) =============================
# Attaches lookup confidence intervals from HIGH to the ML table for QC/plots.
python3 - <<'PY'
import os, pandas as pd
B = os.environ["BASE"]
ml = pd.read_csv(f"{B}/ML_results.with_derived.csv", index_col=0)
look = pd.read_csv(f"{B}/kinetics_from_lookup_HIGH.tsv", sep='\t', index_col=0) if os.path.exists(f"{B}/kinetics_from_lookup_HIGH.tsv") else None

wanted=[]
for base in ["kon","koff","ksyn","burstsize"]:
    for p in [f"{base}_50%conf_low", f"{base}_50%conf_high",
              f"{base}_90%conf_low", f"{base}_90%conf_high",
              f"{base}_95%conf_low", f"{base}_95%conf_high",
              f"{base}_bootstrapmedian", f"{base}_bootstrapfail%"]:
        if look is not None and p in look.columns:
            wanted.append(p)

qc = look[wanted].copy() if look is not None and wanted else pd.DataFrame(index=ml.index)
qc.columns = ["lookup_" + c.replace("burstsize","burst_size") for c in qc.columns]
out = ml.join(qc, how="left")
out.to_csv(f"{B}/ML_results.with_derived_QC.csv")
print("Wrote", f"{B}/ML_results.with_derived_QC.csv", "rows:", len(out), "QC_cols:", (qc.shape[1] if not qc.empty else 0))
PY

echo ">>> Done."
