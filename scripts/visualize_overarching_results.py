#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import pandas as pd
import plotly.graph_objects as go


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#D55E00",
    "neutral": "#999999",
    "positive": "#009E73",
}
COUNTRY_COLORS = {
    "dutch": "#2C7BB6",
    "german": "#7F3C8D",
    "italian": "#E69F00",
}
MIN_PROVINCE_SENTENCES = 46


class SplitCountrySwatch:
    def __init__(self, color: str) -> None:
        self.color = color


class SplitCountrySwatchHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        light = mpatches.Rectangle(
            (xdescent, ydescent),
            width / 2,
            height,
            facecolor=to_rgba(orig_handle.color, 0.34),
            edgecolor="none",
            transform=trans,
        )
        dark = mpatches.Rectangle(
            (xdescent + width / 2, ydescent),
            width / 2,
            height,
            facecolor=to_rgba(orig_handle.color, 0.92),
            edgecolor="none",
            transform=trans,
        )
        return [light, dark]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--admin-csvs", nargs="+", required=True)
    ap.add_argument("--province-gpkgs", nargs="+", required=True)
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


def normalize_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.'()]", "", text)
    return text.strip()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
        }
    )


def country_color(language: str) -> str:
    if language in COUNTRY_COLORS:
        return COUNTRY_COLORS[language]
    palette = ["#2C7BB6", "#7F3C8D", "#E69F00", "#009E73", "#CC6677"]
    return palette[abs(hash(language)) % len(palette)]


def pick_col_by_regex(cols: list[str], patterns: list[str]) -> str | None:
    cols_l = [c.lower() for c in cols]
    for pat in patterns:
        for col, col_l in zip(cols, cols_l, strict=False):
            if re.search(pat, col_l):
                return col
    return None


def pick_admin_col(cols: list[str], exact_names: list[str], patterns: list[str]) -> str | None:
    cols_l = [c.lower() for c in cols]
    exact_l = [name.lower() for name in exact_names]
    for target in exact_l:
        for col, col_l in zip(cols, cols_l, strict=False):
            if col_l == target:
                return col
    return pick_col_by_regex(cols, patterns)


def pick_province_name_col(cols: list[str]) -> str | None:
    return pick_admin_col(
        cols,
        exact_names=["statnaam", "prov_name", "name", "gen"],
        patterns=[r"provincie.*naam", r"prov.*name", r"\bnaam\b", r"name"],
    )


def read_admin_tables(
    languages: list[str],
    countries: list[str],
    admin_csvs: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for language, country, path in zip(languages, countries, admin_csvs, strict=True):
        df = pd.read_csv(path)
        sentiment_col = "sentiment_norm" if "sentiment_norm" in df.columns else "sentiment"
        if sentiment_col not in df.columns:
            raise ValueError(f"{path} is missing a sentiment column.")

        df["_sent"] = normalize_sentiment(df[sentiment_col])
        df = df[df["_sent"].isin(SENTIMENT_ORDER)].copy()
        df["language"] = language
        df["country"] = country
        df["country_color"] = country_color(language)
        if "province_name" not in df.columns:
            df["province_name"] = pd.NA
        frames.append(df)

    if not frames:
        raise ValueError("No administrative CSVs were provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def counts_to_summary(counts: pd.DataFrame) -> pd.DataFrame:
    counts = counts.reindex(columns=SENTIMENT_ORDER, fill_value=0)
    counts = counts.rename(columns={"negative": "n_neg", "neutral": "n_neu", "positive": "n_pos"})
    counts["n_text_units"] = counts["n_neg"] + counts["n_neu"] + counts["n_pos"]
    denom = counts["n_text_units"].replace({0: pd.NA})
    counts["pct_neg"] = 100 * counts["n_neg"] / denom
    counts["pct_neu"] = 100 * counts["n_neu"] / denom
    counts["pct_pos"] = 100 * counts["n_pos"] / denom
    counts["polarity_balance"] = counts["pct_pos"] - counts["pct_neg"]
    return counts


def build_province_sentiment_table(admin_df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    df = admin_df.copy()
    has_province = df["province_name"].notna() & df["province_name"].astype(str).str.strip().ne("")
    province_df = df.loc[has_province].copy()
    province_df["province_name"] = province_df["province_name"].astype(str).str.strip()

    province_counts = (
        province_df.groupby(["language", "country", "country_color", "province_name", "_sent"])
        .size()
        .unstack(fill_value=0)
    )
    province_summary = counts_to_summary(province_counts).reset_index()
    province_summary = province_summary[province_summary["n_text_units"] >= MIN_PROVINCE_SENTENCES].copy()
    province_summary["is_country_average"] = False
    province_summary["display_name"] = province_summary["province_name"]

    average_counts = (
        df.groupby(["language", "country", "country_color", "_sent"])
        .size()
        .unstack(fill_value=0)
    )
    average_summary = counts_to_summary(average_counts).reset_index()
    average_summary["province_name"] = "Country average"
    average_summary["is_country_average"] = True
    average_summary["display_name"] = "Average"

    out = pd.concat([average_summary, province_summary], ignore_index=True, sort=False)
    language_order = {language: idx for idx, language in enumerate(languages)}
    out["_language_order"] = out["language"].map(language_order).fillna(len(language_order)).astype(int)
    out["_average_order"] = out["is_country_average"].map({True: 0, False: 1}).astype(int)
    out = out.sort_values(
        ["_language_order", "_average_order", "polarity_balance", "n_text_units", "province_name"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    return out.drop(columns=["_language_order", "_average_order"])


def plot_province_balance(province_tbl: pd.DataFrame, out_path: Path) -> None:
    plot_df = province_tbl.copy().reset_index(drop=True)
    configure_plot_style()

    width = max(12, 0.34 * len(plot_df) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 6.8), dpi=300)
    x = range(len(plot_df))

    for idx, row in plot_df.iterrows():
        color = row["country_color"]
        ax.bar(idx, -row["pct_neg"], color=color, alpha=0.34, edgecolor="white", linewidth=0.7)
        ax.bar(idx, row["pct_pos"], color=color, alpha=0.92, edgecolor="white", linewidth=0.7)

    bar_extent = float(plot_df[["pct_neg", "pct_pos"]].abs().max().max())
    balance_extent = float(plot_df["polarity_balance"].abs().max())
    base_extent = max(60, bar_extent, balance_extent)
    balance_label_offset = max(3.0, base_extent * 0.04)
    label_cushion = max(8.0, base_extent * 0.1)
    y_limit = max(60, bar_extent + label_cushion, balance_extent + balance_label_offset + label_cushion)
    ax.scatter(
        list(x),
        plot_df["polarity_balance"],
        marker="D",
        s=22,
        color="#1f1f1f",
        edgecolor="white",
        linewidth=0.4,
        zorder=4,
        label="Balance",
    )
    for idx, balance in enumerate(plot_df["polarity_balance"]):
        if pd.isna(balance):
            continue
        ax.text(
            idx,
            balance + (balance_label_offset if balance >= 0 else -balance_label_offset),
            f"{balance:+.0f}%",
            ha="center",
            va="bottom" if balance >= 0 else "top",
            fontsize=8,
            color="#1f1f1f",
            rotation=90,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="none", alpha=0.82),
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Percentage of sentences (%)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{abs(y):.0f}%"))
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlim(-0.75, len(plot_df) - 0.25)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["display_name"], rotation=60, ha="right")

    for idx, total in enumerate(plot_df["n_text_units"]):
        ax.text(
            idx,
            y_limit * 1.02,
            f"n={int(total)}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#3a3a3a",
            rotation=90,
            clip_on=False,
        )

    for _, group in plot_df.groupby("country", sort=False):
        start = int(group.index.min())
        if start > 0:
            ax.axvline(start - 0.5, color="#cfcfcf", linewidth=0.8)

    handles = [
        SplitCountrySwatch(row["country_color"])
        for _, row in plot_df[["country", "country_color"]].drop_duplicates().iterrows()
    ]
    labels = [row["country"] for _, row in plot_df[["country", "country_color"]].drop_duplicates().iterrows()]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#1f1f1f",
            markeredgecolor="white",
            markersize=6,
            label="Balance",
        )
    )
    labels.append("Balance")
    ax.legend(
        handles=handles,
        labels=labels,
        handler_map={SplitCountrySwatch: SplitCountrySwatchHandler()},
        frameon=False,
        ncol=min(4, len(handles)),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.grid(False)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_semicolon_values(value: object) -> list[str]:
    if isinstance(value, list):
        values = value
    elif pd.isna(value):
        values = []
    else:
        values = str(value).split(";")
    return [str(v).strip() for v in values if str(v).strip()]


def build_frame_sentiment_table(admin_df: pd.DataFrame) -> pd.DataFrame:
    if "matched_categories_str" not in admin_df.columns:
        raise ValueError("Combined admin data is missing matched_categories_str.")

    overall_counts = admin_df["_sent"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    overall = counts_to_summary(pd.DataFrame([overall_counts], index=["All analyzed sentences"]))

    frame_df = admin_df[["matched_categories_str", "_sent"]].copy()
    frame_df["frame"] = frame_df["matched_categories_str"].apply(parse_semicolon_values)
    frame_df = frame_df.explode("frame")
    frame_df["frame"] = frame_df["frame"].astype(str).str.strip()
    frame_df = frame_df[frame_df["frame"].ne("")]

    frame_counts = frame_df.groupby(["frame", "_sent"]).size().unstack(fill_value=0)
    frame_summary = counts_to_summary(frame_counts)
    frame_summary = frame_summary.sort_values("n_text_units", ascending=False)

    out = pd.concat([overall, frame_summary], axis=0)
    out.index.name = "frame"
    return out.reset_index()


def plot_frame_sentiment_distribution(frame_tbl: pd.DataFrame, out_path: Path) -> None:
    plot_df = frame_tbl.copy()
    plot_df = plot_df.set_index("frame")
    plot_data = plot_df[["pct_neg", "pct_neu", "pct_pos"]].rename(
        columns={"pct_neg": "negative", "pct_neu": "neutral", "pct_pos": "positive"}
    )

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=300)
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

    ax.set_ylabel("Percentage of sentences (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    for idx, total in enumerate(plot_df["n_text_units"]):
        ax.text(idx, 101.5, f"n={int(total)}", ha="center", va="bottom", fontsize=9, rotation=90, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.grid(False)
    plt.xticks(rotation=45, ha="right")
    fig.legend(
        *ax.get_legend_handles_labels(),
        title="Sentiment",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def explode_frame_sentiments(admin_df: pd.DataFrame) -> pd.DataFrame:
    if "matched_categories_str" not in admin_df.columns:
        raise ValueError("Combined admin data is missing matched_categories_str.")

    frame_df = admin_df[["language", "country", "country_color", "matched_categories_str", "_sent"]].copy()
    frame_df["frame"] = frame_df["matched_categories_str"].apply(parse_semicolon_values)
    frame_df = frame_df.explode("frame")
    frame_df["frame"] = frame_df["frame"].astype(str).str.strip()
    return frame_df[frame_df["frame"].ne("")].copy()


def build_country_frame_balance_table(admin_df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    frame_df = explode_frame_sentiments(admin_df)
    frame_counts = (
        frame_df.groupby(["language", "country", "country_color", "frame", "_sent"])
        .size()
        .unstack(fill_value=0)
    )
    frame_summary = counts_to_summary(frame_counts).reset_index()

    overall_counts = frame_df.groupby(["frame", "_sent"]).size().unstack(fill_value=0)
    overall_summary = counts_to_summary(overall_counts)
    frame_order = (
        overall_summary.sort_values(["polarity_balance", "n_text_units"], ascending=[True, False])
        .index.astype(str)
        .tolist()
    )
    language_order = {language: idx for idx, language in enumerate(languages)}
    frame_rank = {frame: idx for idx, frame in enumerate(frame_order)}
    frame_summary["_language_order"] = frame_summary["language"].map(language_order).fillna(len(language_order)).astype(int)
    frame_summary["_frame_order"] = frame_summary["frame"].map(frame_rank).fillna(len(frame_rank)).astype(int)
    frame_summary = frame_summary.sort_values(["_frame_order", "_language_order"]).reset_index(drop=True)
    return frame_summary.drop(columns=["_language_order", "_frame_order"])


def plot_country_frame_balance(country_frame_tbl: pd.DataFrame, out_path: Path) -> None:
    if country_frame_tbl.empty:
        raise ValueError("No country-frame sentiment balance rows were available to plot.")

    country_meta = country_frame_tbl[["country", "country_color"]].drop_duplicates().reset_index(drop=True)
    frames = country_frame_tbl["frame"].drop_duplicates().tolist()
    pivot = country_frame_tbl.pivot_table(index="frame", columns="country", values="polarity_balance", aggfunc="first")
    pivot = pivot.reindex(index=frames, columns=country_meta["country"])

    configure_plot_style()
    fig_height = max(4.8, 0.74 * len(frames) + 1.8)
    fig, ax = plt.subplots(figsize=(9.8, fig_height), dpi=300)

    y_positions = list(range(len(frames)))
    bar_height = min(0.2, 0.66 / max(len(country_meta), 1))
    center_offset = (len(country_meta) - 1) / 2
    max_abs = float(country_frame_tbl["polarity_balance"].abs().max())
    x_limit = max(60, max_abs + 14)

    for y in [position + 0.5 for position in y_positions[:-1]]:
        ax.hlines(y, -x_limit, x_limit, color="#e7e7e7", linewidth=0.8, zorder=0)

    label_pad = max(1.4, x_limit * 0.018)

    for country_idx, row in country_meta.iterrows():
        country = row["country"]
        values = pivot[country]
        y_offsets = [y + (country_idx - center_offset) * bar_height for y in y_positions]
        ax.barh(
            y_offsets,
            values,
            height=bar_height,
            label=country,
            color=row["country_color"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        for value, y_offset in zip(values, y_offsets, strict=False):
            if pd.isna(value):
                continue
            ax.text(
                value + (label_pad if value >= 0 else -label_pad),
                y_offset,
                f"{value:+.0f}%",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
                color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
                zorder=4,
            )

    ax.set_xlim(-x_limit, x_limit)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frames)
    ax.set_xlabel("Sentiment balance (positive - negative, percentage points)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(False)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    country_handles = [
        mpatches.Patch(facecolor=row["country_color"], edgecolor="white", alpha=0.88, label=row["country"])
        for _, row in country_meta.iterrows()
    ]
    fig.legend(
        handles=country_handles,
        frameon=False,
        ncol=min(3, len(country_handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def load_province_polygons(
    languages: list[str],
    countries: list[str],
    province_gpkgs: list[str],
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for language, country, path in zip(languages, countries, province_gpkgs, strict=True):
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path} has no CRS.")
        name_col = pick_province_name_col(list(gdf.columns))
        if name_col is None:
            raise ValueError(f"Could not detect province name column in {path}.")

        gdf = gdf[[name_col, "geometry"]].rename(columns={name_col: "province_name"}).copy()
        gdf["province_name"] = gdf["province_name"].astype(str).str.strip()
        gdf = gdf[gdf["province_name"].ne("")]
        gdf = gdf.to_crs("EPSG:4326")
        gdf["language"] = language
        gdf["country"] = country
        gdf["_province_key"] = gdf["province_name"].map(normalize_key)
        gdf = gdf.dissolve(by=["language", "country", "_province_key", "province_name"], as_index=False)
        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=["language", "country", "province_name", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True, sort=False), geometry="geometry", crs="EPSG:4326")


def build_map_polygons(
    province_tbl: pd.DataFrame,
    province_polygons: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    province_scores = province_tbl[~province_tbl["is_country_average"]].copy()
    province_scores["_province_key"] = province_scores["province_name"].map(normalize_key)

    merged = province_polygons.merge(
        province_scores[
            [
                "language",
                "country",
                "_province_key",
                "n_text_units",
                "pct_neg",
                "pct_neu",
                "pct_pos",
                "polarity_balance",
            ]
        ],
        on=["language", "country", "_province_key"],
        how="inner",
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged["geometry"] = merged.geometry.simplify(0.01, preserve_topology=True)
    merged = merged.reset_index(drop=True)
    merged["_feature_id"] = merged.index.astype(str)
    return merged


def build_map_points(admin_df: pd.DataFrame, map_polygons: gpd.GeoDataFrame) -> pd.DataFrame:
    if not {"lon", "lat"}.issubset(admin_df.columns):
        return pd.DataFrame(columns=["lon", "lat", "sentiment"])
    points = admin_df[["lon", "lat", "_sent", "language", "country", "province_name"]].copy()
    points["province_name"] = points["province_name"].astype(str).str.strip()
    points["_province_key"] = points["province_name"].map(normalize_key)
    included_provinces = map_polygons[["language", "country", "_province_key"]].drop_duplicates()
    points = points.merge(included_provinces, on=["language", "country", "_province_key"], how="inner")
    points["lon"] = pd.to_numeric(points["lon"], errors="coerce")
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    return points.dropna(subset=["lon", "lat"]).copy()


def plot_static_sentiment_map(
    polygons: gpd.GeoDataFrame,
    points: pd.DataFrame,
    out_path: Path,
) -> None:
    if polygons.empty:
        raise ValueError("No province polygons matched the cross-language sentiment table.")

    polygons_json = json.loads(polygons.to_json(default=str))
    hover = [
        (
            f"{row.province_name}<br>{row.country}<br>"
            f"Balance: {row.polarity_balance:.1f} pp<br>"
            f"Positive: {row.pct_pos:.1f}%<br>"
            f"Neutral: {row.pct_neu:.1f}%<br>"
            f"Negative: {row.pct_neg:.1f}%<br>"
            f"n={int(row.n_text_units)}"
        )
        for row in polygons.itertuples(index=False)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=polygons_json,
            featureidkey="properties._feature_id",
            locations=polygons["_feature_id"],
            z=polygons["polarity_balance"],
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[
                [0.0, "#8C510A"],
                [0.5, "#F5F5F5"],
                [1.0, "#01665E"],
            ],
            zmid=0,
            marker_line_color="#ffffff",
            marker_line_width=0.45,
            colorbar=dict(title="Balance", ticksuffix=" pp", len=0.64),
            name="Province sentiment balance",
        )
    )

    if not points.empty:
        fig.add_trace(
            go.Scattergeo(
                lon=points["lon"],
                lat=points["lat"],
                mode="markers",
                marker=dict(size=2.2, color="#202020", opacity=0.18, line=dict(color="rgba(255,255,255,0.35)", width=0.1)),
                hoverinfo="skip",
                name="Geocoded sentence locations",
            )
        )

    fig.update_geos(
        scope="europe",
        projection_type="natural earth",
        showland=True,
        landcolor="#F3EFE6",
        showocean=True,
        oceancolor="#DCEAF2",
        showlakes=True,
        lakecolor="#DCEAF2",
        showrivers=True,
        rivercolor="#C9DCE8",
        showcountries=True,
        countrycolor="#AFAFAF",
        countrywidth=0.7,
        showcoastlines=True,
        coastlinecolor="#8F8F8F",
        coastlinewidth=0.7,
        bgcolor="#F8F4EC",
        lonaxis_range=[-12, 35],
        lataxis_range=[35, 58],
    )
    fig.update_layout(
        width=1400,
        height=980,
        paper_bgcolor="#F8F4EC",
        plot_bgcolor="#F8F4EC",
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02, bgcolor="rgba(248,244,236,0.72)"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(out_path, scale=2)
    except ValueError as exc:
        raise RuntimeError("Writing the static map requires kaleido. Install the configured kaleido dependency and rerun.") from exc


def validate_args(args: argparse.Namespace) -> None:
    lengths = {
        "languages": len(args.languages),
        "countries": len(args.countries),
        "admin_csvs": len(args.admin_csvs),
        "province_gpkgs": len(args.province_gpkgs),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Expected equal argument counts, got {lengths}.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    admin_df = read_admin_tables(args.languages, args.countries, args.admin_csvs)

    province_tbl = build_province_sentiment_table(admin_df, args.languages)
    province_tbl.to_csv(output_dir / "all_languages_province_sentiment_table.csv", index=False)
    plot_province_balance(province_tbl, output_dir / "all_languages_province_sentiment_balance.png")

    frame_tbl = build_frame_sentiment_table(admin_df)
    frame_tbl.to_csv(output_dir / "all_languages_frames_sentiment_table.csv", index=False)
    plot_frame_sentiment_distribution(frame_tbl, output_dir / "all_languages_frames_sentiment_distribution.png")

    country_frame_tbl = build_country_frame_balance_table(admin_df, args.languages)
    country_frame_tbl.to_csv(output_dir / "all_languages_frames_country_sentiment_balance_table.csv", index=False)
    plot_country_frame_balance(
        country_frame_tbl,
        output_dir / "all_languages_frames_country_sentiment_balance.png",
    )

    province_polygons = load_province_polygons(args.languages, args.countries, args.province_gpkgs)
    map_polygons = build_map_polygons(province_tbl, province_polygons)
    map_points = build_map_points(admin_df, map_polygons)
    plot_static_sentiment_map(map_polygons, map_points, output_dir / "all_languages_province_sentiment_map.png")

    print("Wrote:", output_dir / "all_languages_province_sentiment_table.csv")
    print("Wrote:", output_dir / "all_languages_province_sentiment_balance.png")
    print("Wrote:", output_dir / "all_languages_frames_sentiment_table.csv")
    print("Wrote:", output_dir / "all_languages_frames_sentiment_distribution.png")
    print("Wrote:", output_dir / "all_languages_frames_country_sentiment_balance_table.csv")
    print("Wrote:", output_dir / "all_languages_frames_country_sentiment_balance.png")
    print("Wrote:", output_dir / "all_languages_province_sentiment_map.png")


if __name__ == "__main__":
    main()
