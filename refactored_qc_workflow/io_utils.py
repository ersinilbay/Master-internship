import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from config import ProjectPaths


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


class OutputManager:
    """Keeps output naming exactly as in the original script, just less chaotic."""

    def __init__(self, output_dir: Path, suffix: str):
        self.output_dir = output_dir
        self.suffix = suffix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_suffix(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        return f"{base}{self.suffix}{ext}"

    def out_path(self, filename: str) -> Path:
        return self.output_dir / filename

    def savefig_named(self, filename: str, **kwargs) -> None:
        plt.savefig(self.out_path(self.add_suffix(filename)), bbox_inches="tight", **kwargs)

    def savefig_paper(self, filename_svg: str) -> None:
        base, ext = os.path.splitext(filename_svg)
        name = f"{base}_paper.svg" if ext.lower() != ".svg" else f"{base}_paper{ext}"
        self.savefig_named(name, format="svg")

    def savefig_svg_for_word(self, filename_svg: str) -> None:
        old = mpl.rcParams["svg.fonttype"]
        mpl.rcParams["svg.fonttype"] = "path"
        self.savefig_named(filename_svg, format="svg")
        mpl.rcParams["svg.fonttype"] = old

    def to_csv_named(self, df: pd.DataFrame, filename: str, **kwargs) -> None:
        df.to_csv(self.out_path(self.add_suffix(filename)), **kwargs)

    def write_h5ad_named(self, adata: sc.AnnData, filename: str, **kwargs) -> None:
        adata.write(self.out_path(self.add_suffix(filename)), **kwargs)


def load_count_tables(paths: ProjectPaths):
    c_df = pd.read_csv(require_file(paths.c_file), sep="\t", index_col=0)
    t_df = pd.read_csv(require_file(paths.t_file), sep="\t", index_col=0)
    return c_df, t_df


def load_slam_reference(paths: ProjectPaths) -> pd.DataFrame:
    slam_df = pd.read_excel(require_file(paths.slam_file), engine="xlrd")
    slam_df["Half-life (h)"] = (
        slam_df["Half-life (h)"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    slam_df["Rsquare"] = (
        slam_df["Rsquare"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    slam = slam_df.loc[slam_df["Rsquare"] > 0.4, ["Name", "Half-life (h)"]].rename(
        columns={"Name": "Gene", "Half-life (h)": "SLAM_half_life_hr"}
    )
    slam = slam.drop_duplicates(subset="Gene")
    return slam


def load_qiu_rate_reference(paths: ProjectPaths) -> pd.DataFrame:
    qiu_df = pd.read_excel(require_file(paths.qiu_file), sheet_name="Supplementary Table 5")
    qiu_df = qiu_df[
        ["Gene", "Degradation_rate_Pluripotent", "Synthesis_rate_Pluripotent"]
    ].dropna()
    qiu_df = qiu_df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return qiu_df


def load_no4su_counts(paths: ProjectPaths) -> pd.DataFrame:
    # Original logic: read gene x cell then transpose to cells x genes
    return pd.read_csv(require_file(paths.no4su_file), sep="\t", index_col=0).T