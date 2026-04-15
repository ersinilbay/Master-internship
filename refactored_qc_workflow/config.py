from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import seaborn as sns

# ==============================
# Project-wide constants
# ==============================

PAPER_CMAP = mpl.cm.viridis
TRANSITION_CMAP = mpl.cm.plasma

PAPER_FIG_DPI = 300
UMAP_LABEL_FS = 7
T_LABEL_HOURS = 4

FIGSIZE = (7.5, 5.5)
INTERACTIVE_DPI = 110

SET1_COLORS = {
    "Pluripotent": "#E41A1C",
    "Intermediate": "#377EB8",
    "2-cell like": "#4DAF4A",
}

MARKERS = ["Nanog", "Zfp42", "Myc", "Zscan4c", "Sp110"]
TWOC_MARKERS = ["Zscan4c", "Sp110", "Dppa2", "Gm4340", "Gm4981"]
PLURI_MARKERS = ["Nanog", "Zfp42", "Myc", "Esrrb"]

TARGET_PROPORTIONS = np.array([98.3, 1.0, 0.7])  # Qiu: [Pluri, Intermediate, 2C-like]
INTER_CUTOFF = 0.005
C2C_CUTOFF = 0.819

STATE_CATEGORIES = ["Pluripotent", "Intermediate", "2-cell like"]
STATE_SLUG = {
    "Pluripotent": "pluripotent",
    "Intermediate": "intermediate",
    "2-cell like": "2cell_like",
}


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    output_dir_name: str = "_rep1_fix"
    suffix: str = "_rep1_fix"

    c_file_name: str = "mESC-WT-rep1_C.txt"
    t_file_name: str = "mESC-WT-rep1_T.txt"
    slam_file_name: str = "41592_2017_BFnmeth4435_MOESM4_ESM.xls"
    qiu_file_name: str = "scNTseq_params.xlsx"
    no4su_file_name: str = "GSM4671630_CK-TFEA-run1n2_ds3_gene_exonic.intronic_tagged.dge.txt"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results" / self.output_dir_name

    @property
    def c_file(self) -> Path:
        return self.data_dir / self.c_file_name

    @property
    def t_file(self) -> Path:
        return self.data_dir / self.t_file_name

    @property
    def slam_file(self) -> Path:
        return self.data_dir / self.slam_file_name

    @property
    def qiu_file(self) -> Path:
        return self.data_dir / self.qiu_file_name

    @property
    def no4su_file(self) -> Path:
        return self.data_dir / self.no4su_file_name


def apply_plot_styles() -> None:
    """Apply the same plotting style choices used in the original script."""
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42

    mpl.rcParams.update(
        {
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
        }
    )

    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": INTERACTIVE_DPI,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    sns.set_context("notebook", font_scale=1.0)