#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from html import escape
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

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
    ap.add_argument("--admin-csv", type=str, default="output/text/sentences_with_categories_admin.csv")
    ap.add_argument("--categories-csv", type=str, default="output/text/sentences_with_categories_short.csv")
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


def text_unit_label(df: pd.DataFrame) -> str:
    return "sentences" if "sentence_text" in df.columns else "paragraphs"


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

    grouped["n_text_units"] = grouped["n_neg"] + grouped["n_neu"] + grouped["n_pos"]
    denom = grouped["n_text_units"].replace({0: pd.NA})
    grouped["pct_neg"] = 100 * grouped["n_neg"] / denom
    grouped["pct_neu"] = 100 * grouped["n_neu"] / denom
    grouped["pct_pos"] = 100 * grouped["n_pos"] / denom
    grouped["polarity_balance"] = grouped["pct_pos"] - grouped["pct_neg"]
    grouped = grouped.sort_values(["n_text_units", "province_name"], ascending=[False, True]).reset_index(drop=True)
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
    ax.set_xlabel("Percentage of sentences")
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
    ax.set_ylabel("Percentage of sentences")
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
    unit_label = text_unit_label(categories_df)

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
    all_label = f"All {unit_label}"
    plot_data.loc[all_label] = overall_percent
    plot_data = plot_data.loc[[all_label] + [idx for idx in plot_data.index if idx != all_label]]

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

    ax.set_ylabel("Percentage of sentences")
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


def truncate_text(text: str, limit: int = 220) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def plot_locations_interactive(
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
        raise ValueError("No valid lon/lat rows available for interactive location map.")

    text_col = "sentence_text" if "sentence_text" in points_df.columns else "paragraph_text"
    if text_col not in points_df.columns:
        raise ValueError("Expected either 'sentence_text' or 'paragraph_text' in administrative CSV.")

    points_df["matched_location_display"] = (
        points_df.get("geo_name_matched", pd.Series(index=points_df.index, dtype=object))
        .fillna(points_df.get("llm_location", pd.Series(index=points_df.index, dtype=object)))
        .fillna("Unknown location")
        .astype(str)
    )
    points_df["text_display"] = points_df[text_col].fillna("").astype(str)
    points_df["text_preview"] = points_df["text_display"].map(truncate_text)
    points_df["sentiment_display"] = (
        normalize_sentiment(points_df["sentiment"])
        if "sentiment" in points_df.columns
        else pd.Series([""] * len(points_df), index=points_df.index)
    ).fillna("")
    points_df["frame_display"] = points_df.get("matched_categories_str", pd.Series(index=points_df.index, dtype=object)).fillna("").astype(str)

    provinces = gpd.read_file(province_gpkg)
    if provinces.crs is None:
        raise ValueError("Province GeoPackage must have a CRS.")
    provinces = provinces.to_crs("EPSG:4326")
    provinces = provinces.reset_index(drop=True).copy()
    provinces["_feature_id"] = provinces.index.astype(str)
    provinces_json = json.loads(provinces.to_json())

    marker_colors = {
        "negative": "#D55E00",
        "neutral": "#999999",
        "positive": "#009E73",
        "": "#c75b39",
    }
    points_df["marker_color"] = points_df["sentiment_display"].map(marker_colors).fillna("#c75b39")

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=provinces_json,
            featureidkey="properties._feature_id",
            locations=provinces["_feature_id"],
            z=[1] * len(provinces),
            colorscale=[[0, "#efe5d2"], [1, "#efe5d2"]],
            showscale=False,
            marker_line_color="#ffffff",
            marker_line_width=1.1,
            hoverinfo="skip",
            name="Provinces",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=points_df["lon"],
            lat=points_df["lat"],
            mode="markers",
            marker=dict(
                size=8,
                color=points_df["marker_color"],
                opacity=0.72,
                line=dict(color="#ffffff", width=0.6),
            ),
            customdata=points_df[
                ["matched_location_display", "text_display", "frame_display", "sentiment_display", "text_preview"]
            ].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Sentiment: %{customdata[3]}<br>"
                "Frames: %{customdata[2]}<br>"
                "%{customdata[4]}"
                "<extra></extra>"
            ),
            name="Matched locations",
        )
    )

    xmin, ymin, xmax, ymax = provinces.total_bounds
    xpad = (xmax - xmin) * 0.04
    ypad = (ymax - ymin) * 0.04

    fig.update_geos(
        fitbounds=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        bgcolor="#f6f1e8",
        lonaxis_range=[xmin - xpad, xmax + xpad],
        lataxis_range=[ymin - ypad, ymax + ypad],
        projection_type="mercator",
    )
    fig.update_layout(
        title="Interactive matched locations",
        paper_bgcolor="#f6f1e8",
        plot_bgcolor="#f6f1e8",
        margin=dict(l=20, r=20, t=60, b=20),
        height=860,
    )

    plot_div_id = "location-map"
    details_div_id = "location-details"
    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=plot_div_id,
        post_script=f"""
const plot = document.getElementById('{plot_div_id}');
const details = document.getElementById('{details_div_id}');
if (plot) {{
  plot.on('plotly_click', function(event) {{
    const point = event.points && event.points[0];
    if (!point || !point.customdata) return;
    const location = point.customdata[0] || 'Unknown location';
    const text = point.customdata[1] || '';
    const frame = point.customdata[2] || '';
    const sentiment = point.customdata[3] || '';
    details.innerHTML = `
      <h3>${{location}}</h3>
      <p><strong>Sentiment:</strong> ${{sentiment || 'n/a'}}</p>
      <p><strong>Frames:</strong> ${{frame || 'n/a'}}</p>
      <p>${{text}}</p>
    `;
  }});
}}
""",
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interactive matched locations</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f6f1e8;
      color: #3d342b;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 2.2fr) minmax(320px, 1fr);
      gap: 18px;
      padding: 18px;
      align-items: start;
    }}
    .panel {{
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(83,72,61,0.18);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(61,52,43,0.08);
      overflow: hidden;
    }}
    .details {{
      padding: 18px 20px;
      position: sticky;
      top: 18px;
      min-height: 200px;
    }}
    .details h2, .details h3 {{
      margin: 0 0 12px 0;
      font-weight: 600;
    }}
    .details p {{
      margin: 0 0 12px 0;
      line-height: 1.5;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .details {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="panel">{fig_html}</div>
    <div class="panel details" id="{details_div_id}">
      <h2>Matched text</h2>
      <p>Hover over a point to inspect the matched location. Click a point to load the full text here.</p>
    </div>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


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
    plot_locations_interactive(
        admin_df,
        province_gpkg,
        output_dir / "locations_map.html",
    )

    print("Wrote:", output_dir / "province_sentiment_table.csv")
    print("Wrote:", output_dir / "provinces_sentiment_balance.png")
    print("Wrote:", output_dir / "provinces_sentiment_distribution.png")
    print("Wrote:", output_dir / "categories_sentiment_distribution.png")
    print("Wrote:", output_dir / "locations_map.html")


if __name__ == "__main__":
    main()
