#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DESCRIPTIVE_BLUE = "#3f6f8f"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--articles-csv", type=str, default="output/workflow/articles_cleaned.csv")
    ap.add_argument("--output-dir", type=str, default="output/figures")
    ap.add_argument("--table-dir", type=str, default="output/text")
    ap.add_argument("--top-n-newspapers", type=int, default=6)
    return ap.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_articles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Articles CSV is empty: {path}")
    return df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()


def add_publish_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_candidates = ["publish_date", "date_parsed", "date", "date_translated"]
    date_col = next((col for col in date_candidates if col in df.columns), None)
    if date_col is None:
        raise ValueError(
            "Expected one of these date columns in articles CSV: "
            f"{date_candidates}"
        )

    dates = pd.to_datetime(df[date_col], errors="coerce")
    df["publish_year"] = dates.dt.year.astype("Int64")
    return df


def build_articles_per_year(df: pd.DataFrame) -> pd.DataFrame:
    if "publish_year" not in df.columns:
        df = add_publish_year(df)

    counts = (
        df.dropna(subset=["publish_year"])
        .groupby("publish_year")
        .size()
        .rename("n_articles")
        .reset_index()
        .sort_values("publish_year")
    )
    counts["publish_year"] = counts["publish_year"].astype(int)
    return counts


def build_top_newspapers(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    newspaper_col = "newspaper" if "newspaper" in df.columns else "source"
    if newspaper_col not in df.columns:
        raise ValueError("Expected either 'newspaper' or 'source' column in articles CSV.")

    newspapers = df[newspaper_col].fillna("").astype(str).str.strip()
    newspapers = newspapers[newspapers.ne("")]
    counts = newspapers.value_counts().head(top_n).rename_axis("newspaper").reset_index(name="n_articles")
    return counts


def plot_articles_per_year(year_counts: pd.DataFrame, out_path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    ax.bar(
        year_counts["publish_year"].astype(str),
        year_counts["n_articles"],
        color=DESCRIPTIVE_BLUE,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of articles")
    ax.set_title("Articles per year")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    for idx, value in enumerate(year_counts["n_articles"]):
        ax.text(idx, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_newspapers(newspaper_counts: pd.DataFrame, out_path: Path) -> None:
    configure_plot_style()
    plot_df = newspaper_counts.sort_values("n_articles", ascending=True)
    fig_height = max(4.2, 0.55 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(8, fig_height), dpi=300)

    ax.barh(
        plot_df["newspaper"],
        plot_df["n_articles"],
        color=DESCRIPTIVE_BLUE,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Number of articles")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(newspaper_counts)} newspapers by article count")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    xmax = plot_df["n_articles"].max()
    ax.set_xlim(0, xmax * 1.12 if xmax else 1)
    for newspaper, value in zip(plot_df["newspaper"], plot_df["n_articles"], strict=False):
        ax.text(value + xmax * 0.015, newspaper, f"{int(value)}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    dated = add_publish_year(df)
    years = dated["publish_year"].dropna()
    newspaper_col = "newspaper" if "newspaper" in df.columns else "source" if "source" in df.columns else None
    unique_newspapers = (
        int(df[newspaper_col].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
        if newspaper_col
        else pd.NA
    )
    summary = pd.DataFrame(
        [
            {"metric": "n_articles", "value": int(len(df))},
            {"metric": "n_articles_with_publish_year", "value": int(years.shape[0])},
            {"metric": "first_publish_year", "value": int(years.min()) if not years.empty else pd.NA},
            {"metric": "last_publish_year", "value": int(years.max()) if not years.empty else pd.NA},
            {"metric": "n_newspapers", "value": unique_newspapers},
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    articles_csv = Path(args.articles_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    table_dir = Path(args.table_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    articles_df = add_publish_year(load_articles(articles_csv))
    year_counts = build_articles_per_year(articles_df)
    newspaper_counts = build_top_newspapers(articles_df, args.top_n_newspapers)

    year_counts.to_csv(table_dir / "articles_per_year.csv", index=False)
    newspaper_counts.to_csv(table_dir / "top_newspapers.csv", index=False)
    write_summary(articles_df, table_dir / "article_descriptives_summary.csv")

    plot_articles_per_year(year_counts, output_dir / "articles_per_year.png")
    plot_top_newspapers(newspaper_counts, output_dir / "top_newspapers.png")

    print("Wrote:", table_dir / "articles_per_year.csv")
    print("Wrote:", output_dir / "articles_per_year.png")
    print("Wrote:", table_dir / "top_newspapers.csv")
    print("Wrote:", output_dir / "top_newspapers.png")
    print("Wrote:", table_dir / "article_descriptives_summary.csv")


if __name__ == "__main__":
    main()
