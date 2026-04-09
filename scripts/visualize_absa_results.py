#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#D55E00",
    "neutral": "#999999",
    "positive": "#009E73",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--admin-csv", type=str, default="output/text/paragraphs_with_categories_admin.csv")
    ap.add_argument("--categories-csv", type=str, default="output/text/paragraphs_with_categories_short.csv")
    ap.add_argument("--province-gpkg", type=str, default="data/dutch/admin_areas_provinces_2025.gpkg")
    ap.add_argument("--output-dir", type=str, default="output/figures")
    return ap.parse_args()


def normalize_sentiment(series: pd.Series) -> pd.Series:
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "neutral/uncertain": "neutral",
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
        }
    )


def build_province_summary(admin_df: pd.DataFrame) -> pd.DataFrame:
    df = admin_df.copy()
    if "province_name" not in df.columns:
        raise ValueError("Expected 'province_name' column in administrative CSV.")

    sentiment_source_col = "sentiment_norm" if "sentiment_norm" in df.columns else "sentiment"
    if sentiment_source_col not in df.columns:
        raise ValueError("Expected either 'sentiment_norm' or 'sentiment' column in administrative CSV.")

    df["_sent"] = normalize_sentiment(df[sentiment_source_col])
    df = df[df["_sent"].isin(SENTIMENT_ORDER)].copy()
    df = df.dropna(subset=["province_name"])
    df = df[df["province_name"].astype(str).str.strip().ne("")]

    grouped = (
        df.groupby(["province_name", "_sent"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
        .rename(columns={"negative": "n_neg", "neutral": "n_neu", "positive": "n_pos"})
        .reset_index()
    )

    grouped["n_paragraphs"] = grouped["n_neg"] + grouped["n_neu"] + grouped["n_pos"]
    denom = grouped["n_paragraphs"].replace({0: pd.NA})
    grouped["pct_neg"] = 100 * grouped["n_neg"] / denom
    grouped["pct_neu"] = 100 * grouped["n_neu"] / denom
    grouped["pct_pos"] = 100 * grouped["n_pos"] / denom
    grouped["polarity_balance"] = grouped["pct_pos"] - grouped["pct_neg"]
    grouped = grouped.sort_values(["n_paragraphs", "province_name"], ascending=[False, True]).reset_index(drop=True)
    return grouped


def plot_province_sentiment_balance(province_tbl: pd.DataFrame, out_path: Path) -> None:
    df = province_tbl.copy().sort_values("polarity_balance")
    configure_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.set_xlim(-60, 60)

    ax.barh(
        df["province_name"],
        -df["pct_neg"],
        label="Negative",
        color=SENTIMENT_COLORS["negative"],
        edgecolor="white",
        linewidth=0.7,
    )
    ax.barh(
        df["province_name"],
        df["pct_pos"],
        label="Positive",
        color=SENTIMENT_COLORS["positive"],
        edgecolor="white",
        linewidth=0.7,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage of paragraphs")
    ax.set_title("Sentiment per province")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.grid(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_province_stacked_distribution(province_tbl: pd.DataFrame, out_path: Path) -> None:
    df = province_tbl.copy().dropna(subset=["province_name"])
    plot_df = (
        df.groupby("province_name", as_index=True)[["n_neg", "n_neu", "n_pos"]]
        .sum()
        .sort_values(["n_neg", "n_neu", "n_pos"], ascending=False)
    )
    plot_df["total"] = plot_df["n_neg"] + plot_df["n_neu"] + plot_df["n_pos"]
    denom = plot_df["total"].replace({0: pd.NA})

    plot_data = pd.DataFrame(index=plot_df.index)
    plot_data["negative"] = 100 * plot_df["n_neg"] / denom
    plot_data["neutral"] = 100 * plot_df["n_neu"] / denom
    plot_data["positive"] = 100 * plot_df["n_pos"] / denom

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bottom = pd.Series(0, index=plot_data.index, dtype=float)

    for sentiment in SENTIMENT_ORDER:
        ax.bar(
            plot_data.index,
            plot_data[sentiment],
            bottom=bottom,
            label=sentiment.capitalize(),
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += plot_data[sentiment]

    ax.set_xlabel("Province")
    ax.set_ylabel("Percentage of paragraphs")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_axisbelow(True)

    for idx, total in enumerate(plot_df["total"]):
        if pd.notna(total):
            ax.text(
                idx,
                101.5,
                f"n={int(total)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#3a3a3a",
                rotation=90,
            )

    ax.set_ylim(0, 108)
    plt.xticks(rotation=45, ha="right")
    ax.legend(
        title="Sentiment",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_category_sentiment_distribution(categories_df: pd.DataFrame, out_path: Path) -> None:
    category_col = "matched_categories_str"
    sentiment_col = "sentiment"
    required = [category_col, sentiment_col]
    missing = [c for c in required if c not in categories_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in categories CSV: {missing}")

    all_sentiments = categories_df[[sentiment_col]].copy().dropna(subset=[sentiment_col])
    all_sentiments["_sent"] = normalize_sentiment(all_sentiments[sentiment_col])
    all_sentiments = all_sentiments[all_sentiments["_sent"].isin(SENTIMENT_ORDER)]
    overall_counts = all_sentiments["_sent"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    overall_percent = (overall_counts / overall_counts.sum()) * 100

    df = categories_df[[category_col, sentiment_col]].copy().dropna(subset=[category_col, sentiment_col])
    df["category"] = df[category_col].astype(str).str.split(";")
    df = df.explode("category")
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"].notna() & df["category"].ne("")]
    df["_sent"] = normalize_sentiment(df[sentiment_col])
    df = df[df["_sent"].isin(SENTIMENT_ORDER)]

    sentiment_counts = (
        df.groupby(["category", "_sent"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    sentiment_counts["total"] = sentiment_counts.sum(axis=1)
    sentiment_counts = sentiment_counts.sort_values("total", ascending=False)
    plot_data = sentiment_counts.drop(columns="total").div(sentiment_counts["total"], axis=0) * 100
    plot_data.loc["All paragraphs"] = overall_percent
    plot_data = plot_data.loc[["All paragraphs"] + [idx for idx in plot_data.index if idx != "All paragraphs"]]

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
    bottom = pd.Series(0, index=plot_data.index, dtype=float)

    for sentiment in SENTIMENT_ORDER:
        ax.bar(
            plot_data.index,
            plot_data[sentiment],
            bottom=bottom,
            label=sentiment.capitalize(),
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += plot_data[sentiment]

    ax.set_ylabel("Percentage of paragraphs")
    ax.set_ylim(0, 100)
    ax.margins(y=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(
        title="Sentiment",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_locations_heatmap(
    admin_df: pd.DataFrame,
    province_gpkg: Path,
    out_path: Path,
) -> None:
    required = {"lon", "lat"}
    missing = required - set(admin_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for location heatmap: {sorted(missing)}")

    points_df = admin_df.copy()
    points_df["lon"] = pd.to_numeric(points_df["lon"], errors="coerce")
    points_df["lat"] = pd.to_numeric(points_df["lat"], errors="coerce")
    points_df = points_df.dropna(subset=["lon", "lat"]).copy()
    if points_df.empty:
        raise ValueError("No valid lon/lat rows available for location heatmap.")

    provinces = gpd.read_file(province_gpkg)
    if provinces.crs is None:
        raise ValueError("Province GeoPackage must have a CRS.")
    provinces = provinces.to_crs("EPSG:4326")
    provinces["_label_point"] = provinces.geometry.representative_point()

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 10.2), dpi=250)
    fig.patch.set_facecolor("#f6f1e8")
    ax.set_facecolor("#f6f1e8")

    provinces.plot(
        ax=ax,
        color="#efe5d2",
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=1.0,
        zorder=0,
    )
    provinces.boundary.plot(ax=ax, color="#53483d", linewidth=0.9, alpha=0.7, zorder=2)

    ax.scatter(
        points_df["lon"],
        points_df["lat"],
        s=16,
        c="#c75b39",
        alpha=0.55,
        edgecolors="white",
        linewidths=0.25,
        zorder=1.5,
    )

    xmin, ymin, xmax, ymax = provinces.total_bounds
    xpad = (xmax - xmin) * 0.04
    ypad = (ymax - ymin) * 0.04
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Plotted locations")
    ax.set_aspect("equal")
    ax.grid(False)

    name_col = None
    for cand in ["statnaam", "prov_name", "name", "naam", "provincie_naam"]:
        if cand in provinces.columns:
            name_col = cand
            break
    if name_col is not None:
        for _, row in provinces.iterrows():
            point = row["_label_point"]
            ax.text(
                point.x,
                point.y,
                str(row[name_col]),
                fontsize=8.5,
                color="#53483d",
                ha="center",
                va="center",
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.18", fc=(246/255, 241/255, 232/255, 0.72), ec="none"),
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    admin_df = pd.read_csv(args.admin_csv)
    categories_df = pd.read_csv(args.categories_csv)
    province_gpkg = Path(args.province_gpkg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    province_tbl = build_province_summary(admin_df)
    province_tbl.to_csv(output_dir / "province_sentiment_table.csv", index=False)

    plot_province_sentiment_balance(
        province_tbl,
        output_dir / "provinces_sentiment_balance.png",
    )
    plot_province_stacked_distribution(
        province_tbl,
        output_dir / "provinces_sentiment_distribution.png",
    )
    plot_category_sentiment_distribution(
        categories_df,
        output_dir / "categories_sentiment_distribution.png",
    )
    plot_locations_heatmap(
        admin_df,
        province_gpkg,
        output_dir / "locations_heatmap.png",
    )

    print("Wrote:", output_dir / "province_sentiment_table.csv")
    print("Wrote:", output_dir / "provinces_sentiment_balance.png")
    print("Wrote:", output_dir / "provinces_sentiment_distribution.png")
    print("Wrote:", output_dir / "categories_sentiment_distribution.png")
    print("Wrote:", output_dir / "locations_heatmap.png")


if __name__ == "__main__":
    main()
