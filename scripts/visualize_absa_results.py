#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from html import escape
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from language_resources import country_aliases, load_keyword_csv, load_location_province_overrides

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1]

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#D55E00",
    "neutral": "#999999",
    "positive": "#009E73",
}
DESCRIPTIVE_BLUE = "#3f6f8f"
MIN_PROVINCE_SENTENCES = 31
LOCATION_PROVINCE_OVERRIDES = {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--admin-csv", type=str, default="output/text/sentences_with_categories_admin.csv")
    ap.add_argument("--categories-csv", type=str, default="output/text/sentences_with_categories_short.csv")
    ap.add_argument("--keywords-csv", type=str, default="vocab/keywords_topics.csv")
    ap.add_argument("--province-gpkg", type=str, default="data/dutch/admin_areas_provinces_2025.gpkg")
    ap.add_argument("--output-dir", type=str, default="output/figures")
    ap.add_argument("--country", type=str, default="Netherlands")
    ap.add_argument("--location-province-overrides", type=str, default="")
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


def parse_semicolon_values(value: object) -> list[str]:
    if isinstance(value, list):
        values = value
    elif pd.isna(value):
        values = []
    else:
        values = str(value).split(";")
    return [str(v).strip() for v in values if str(v).strip()]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "frame"


def normalize_location_value(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_province_overrides(
    df: pd.DataFrame,
    location_province_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if "province_name" not in df.columns:
        return df

    location_cols = [c for c in ["_loc_norm", "_loc_first", "llm_location", "geo_name_matched"] if c in df.columns]
    if not location_cols:
        return df

    combined = pd.Series("", index=df.index, dtype=object)
    for col in location_cols:
        combined = combined + " | " + df[col].map(normalize_location_value)

    overrides = location_province_overrides if location_province_overrides is not None else LOCATION_PROVINCE_OVERRIDES
    for loc_norm, province_name in overrides.items():
        mask = combined.str.contains(rf"(?:^| \| ){re.escape(loc_norm)}(?:$| \| )", regex=True, na=False)
        df.loc[mask, "province_name"] = province_name

    return df


def is_country_location(df: pd.DataFrame, country: str) -> pd.Series:
    location_cols = [c for c in ["_loc_norm", "_loc_first", "llm_location", "geo_name_matched"] if c in df.columns]
    if not location_cols:
        return pd.Series(False, index=df.index)

    aliases = {normalize_location_value(alias) for alias in country_aliases(country)}
    mask = pd.Series(False, index=df.index)
    for col in location_cols:
        mask = mask | df[col].map(normalize_location_value).isin(aliases)
    return mask


def load_frame_keyword_vocab(path: Path) -> tuple[list[str], dict[str, set[str]], dict[str, dict[str, str]]]:
    vocab_df = load_keyword_csv(path)

    frame_order: list[str] = []
    frame_keywords: dict[str, set[str]] = {}
    display_lookup: dict[str, dict[str, str]] = {}

    for col in vocab_df.columns:
        frame = str(col).strip()
        if not frame:
            continue
        keywords = (
            vocab_df[col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        keywords = keywords[keywords.ne("")]
        if frame not in frame_order:
            frame_order.append(frame)
        frame_keywords[frame] = {kw.lower() for kw in keywords}
        display_lookup[frame] = {kw.lower(): kw for kw in keywords}

    return frame_order, frame_keywords, display_lookup


def build_province_summary(
    admin_df: pd.DataFrame,
    country: str,
    location_province_overrides: dict[str, str],
) -> pd.DataFrame:
    df = admin_df.copy()
    if "province_name" not in df.columns:
        raise ValueError("Expected 'province_name' column in administrative CSV.")

    sentiment_source_col = "sentiment_norm" if "sentiment_norm" in df.columns else "sentiment"
    if sentiment_source_col not in df.columns:
        raise ValueError("Expected either 'sentiment_norm' or 'sentiment' column in administrative CSV.")

    df["_sent"] = normalize_sentiment(df[sentiment_source_col])
    df = df[df["_sent"].isin(SENTIMENT_ORDER)].copy()
    df = apply_province_overrides(df, location_province_overrides)

    has_province = df["province_name"].notna() & df["province_name"].astype(str).str.strip().ne("")
    include_in_overall = has_province | (~has_province & is_country_location(df, country))
    overall_counts = df.loc[include_in_overall, "_sent"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    overall_total = int(overall_counts.sum())
    overall = pd.DataFrame(
        [{
            "province_name": "All sentences",
            "n_neg": int(overall_counts["negative"]),
            "n_neu": int(overall_counts["neutral"]),
            "n_pos": int(overall_counts["positive"]),
            "n_text_units": overall_total,
            "pct_neg": 100 * overall_counts["negative"] / overall_total if overall_total else pd.NA,
            "pct_neu": 100 * overall_counts["neutral"] / overall_total if overall_total else pd.NA,
            "pct_pos": 100 * overall_counts["positive"] / overall_total if overall_total else pd.NA,
            "polarity_balance": (
                100 * overall_counts["positive"] / overall_total
                - 100 * overall_counts["negative"] / overall_total
                if overall_total
                else pd.NA
            ),
        }]
    )

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
    grouped = grouped[grouped["n_text_units"] >= MIN_PROVINCE_SENTENCES].copy()
    grouped = grouped.sort_values(["n_text_units", "province_name"], ascending=[False, True]).reset_index(drop=True)
    return pd.concat([overall, grouped], ignore_index=True)


def plot_province_sentiment_balance(province_tbl: pd.DataFrame, out_path: Path) -> None:
    df = province_tbl.copy()
    df = df.sort_values("polarity_balance")
    df["province_label"] = df.apply(
        lambda row: f"{row['province_name']} (n={int(row['n_text_units'])})",
        axis=1,
    )
    configure_plot_style()

    fig, ax = plt.subplots(figsize=(8.4, 6.6), dpi=200)
    x_limit = max(
        60,
        float(df[["pct_neg", "pct_pos", "polarity_balance"]].abs().max().max()) + 8,
    )
    ax.set_xlim(-x_limit, x_limit)

    ax.barh(
        df["province_label"],
        -df["pct_neg"],
        label="Negative",
        color=SENTIMENT_COLORS["negative"],
        edgecolor="white",
        linewidth=0.7,
    )
    ax.barh(
        df["province_label"],
        df["pct_pos"],
        label="Positive",
        color=SENTIMENT_COLORS["positive"],
        edgecolor="white",
        linewidth=0.7,
    )
    ax.scatter(
        df["polarity_balance"],
        df["province_label"],
        marker="D",
        s=28,
        color="#1f1f1f",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="Balance",
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage of sentences; diamond = positive - negative")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.grid(False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_province_stacked_distribution(province_tbl: pd.DataFrame, out_path: Path) -> None:
    df = province_tbl.copy().dropna(subset=["province_name"])
    plot_df = df.set_index("province_name")[["n_neg", "n_neu", "n_pos"]].copy()
    plot_df["total"] = plot_df["n_neg"] + plot_df["n_neu"] + plot_df["n_pos"]
    denom = plot_df["total"].replace({0: pd.NA})

    plot_data = pd.DataFrame(index=plot_df.index)
    plot_data["negative"] = 100 * plot_df["n_neg"] / denom
    plot_data["neutral"] = 100 * plot_df["n_neu"] / denom
    plot_data["positive"] = 100 * plot_df["n_pos"] / denom
    if "All sentences" in plot_data.index:
        ordered_index = ["All sentences"] + [idx for idx in plot_data.index if idx != "All sentences"]
        plot_data = plot_data.loc[ordered_index]
        plot_df = plot_df.loc[ordered_index]

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
                clip_on=False,
            )

    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    fig.legend(
        *ax.get_legend_handles_labels(),
        title="Sentiment",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
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
    plot_totals = sentiment_counts["total"].copy()
    plot_totals.loc[all_label] = overall_counts.sum()
    plot_totals = plot_totals.loc[plot_data.index]

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=300)
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
    for idx, total in enumerate(plot_totals):
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
                clip_on=False,
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
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
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def build_region_frame_counts(
    admin_df: pd.DataFrame,
    province_tbl: pd.DataFrame,
) -> pd.DataFrame:
    required = ["province_name", "matched_categories_str"]
    missing = [c for c in required if c not in admin_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for province-frame plots: {missing}")

    eligible_provinces = set(
        province_tbl.loc[
            province_tbl["province_name"].astype(str).ne("All sentences"),
            "province_name",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    df = admin_df[required].copy().dropna(subset=required)
    df["province_name"] = df["province_name"].astype(str).str.strip()
    df = df[df["province_name"].isin(eligible_provinces)]
    df["frame"] = df["matched_categories_str"].apply(parse_semicolon_values)
    df = df.explode("frame")
    df["frame"] = df["frame"].astype(str).str.strip()
    df = df[df["frame"].ne("")]

    if df.empty:
        return pd.DataFrame(columns=["province_name", "frame", "n_mentions"])

    return (
        df.groupby(["province_name", "frame"])
        .size()
        .rename("n_mentions")
        .reset_index()
        .sort_values(["province_name", "n_mentions", "frame"], ascending=[True, False, True])
    )


def eligible_province_summary(province_tbl: pd.DataFrame) -> pd.DataFrame:
    return province_tbl[
        province_tbl["province_name"].notna()
        & province_tbl["province_name"].astype(str).str.strip().ne("")
        & province_tbl["province_name"].astype(str).ne("All sentences")
    ].copy()


def plot_single_region_frame_counts(region_counts: pd.DataFrame, province_name: str, out_path: Path) -> None:
    plot_df = region_counts.sort_values(["n_mentions", "frame"], ascending=[True, True])

    configure_plot_style()
    fig_height = max(4.2, 0.55 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(8, fig_height), dpi=300)

    ax.barh(
        plot_df["frame"],
        plot_df["n_mentions"],
        color=DESCRIPTIVE_BLUE,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Number of frame mentions")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    xmax = plot_df["n_mentions"].max()
    ax.set_xlim(0, xmax * 1.12 if xmax else 1)
    for frame, value in zip(plot_df["frame"], plot_df["n_mentions"], strict=False):
        ax.text(value + xmax * 0.015, frame, f"{int(value)}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_region_sentiment_extreme_frame_comparison(
    region_counts: pd.DataFrame,
    province_summary: pd.DataFrame,
    frame_order: list[str],
    out_path: Path,
) -> None:
    province_names = most_negative_positive_provinces(province_summary)
    if not province_names:
        return

    compare = build_extreme_province_frame_matrix(region_counts, province_names, frame_order)
    if compare.empty:
        return

    compare["total"] = compare.sum(axis=1)
    compare = compare.sort_values(["total"], ascending=True)
    compare = compare.drop(columns="total")

    configure_plot_style()
    fig_height = max(4.4, 0.58 * len(compare) + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_height), dpi=300)
    y_positions = range(len(compare))
    bar_height = 0.38

    ax.barh(
        [y - bar_height / 2 for y in y_positions],
        compare[province_names[0]],
        height=bar_height,
        label=f"{province_names[0]} (most negative)",
        color=SENTIMENT_COLORS["negative"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.barh(
        [y + bar_height / 2 for y in y_positions],
        compare[province_names[1]],
        height=bar_height,
        label=f"{province_names[1]} (most positive)",
        color=SENTIMENT_COLORS["positive"],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(compare.index)
    ax.set_xlabel("Number of frame mentions")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    xmax = float(compare.max().max())
    ax.set_xlim(0, xmax * 1.18 if xmax else 1)
    offset = xmax * 0.015 if xmax else 0.02
    for idx, (_, row) in enumerate(compare.iterrows()):
        for y_pos, province_name in [
            (idx - bar_height / 2, province_names[0]),
            (idx + bar_height / 2, province_names[1]),
        ]:
            value = row[province_name]
            ax.text(value + offset, y_pos, f"{int(value)}", va="center", ha="left", fontsize=9)

    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def most_negative_positive_provinces(province_summary: pd.DataFrame) -> list[str]:
    if province_summary.empty or province_summary["province_name"].nunique() < 2:
        return []

    negative_region = province_summary.sort_values(
        ["polarity_balance", "n_text_units", "province_name"],
        ascending=[True, False, True],
    ).iloc[0]
    positive_candidates = province_summary[
        province_summary["province_name"].astype(str).ne(str(negative_region["province_name"]))
    ]
    if positive_candidates.empty:
        return []

    positive_region = positive_candidates.sort_values(
        ["polarity_balance", "n_text_units", "province_name"],
        ascending=[False, False, True],
    ).iloc[0]
    return [str(negative_region["province_name"]), str(positive_region["province_name"])]


def build_extreme_province_frame_matrix(
    region_counts: pd.DataFrame,
    province_names: list[str],
    frame_order: list[str],
) -> pd.DataFrame:
    if region_counts.empty or len(province_names) != 2:
        return pd.DataFrame()

    compare = (
        region_counts[region_counts["province_name"].isin(province_names)]
        .pivot_table(index="frame", columns="province_name", values="n_mentions", aggfunc="sum", fill_value=0)
    )
    compare = compare.reindex(columns=province_names, fill_value=0)
    ordered_frames = [frame for frame in frame_order if frame in compare.index]
    extra_frames = [frame for frame in compare.index if frame not in ordered_frames]
    return compare.loc[ordered_frames + sorted(extra_frames)]


def plot_region_sentiment_extreme_frame_relative_importance(
    region_counts: pd.DataFrame,
    province_summary: pd.DataFrame,
    frame_order: list[str],
    out_path: Path,
) -> None:
    province_names = most_negative_positive_provinces(province_summary)
    if not province_names:
        return

    compare = build_extreme_province_frame_matrix(region_counts, province_names, frame_order)
    if compare.empty:
        return

    denom = compare.sum(axis=0).replace({0: pd.NA})
    plot_data = (compare.div(denom, axis=1) * 100).fillna(0)
    plot_data["total"] = plot_data.sum(axis=1)
    plot_data = plot_data.sort_values(["total"], ascending=True)
    plot_data = plot_data.drop(columns="total")
    if plot_data.empty:
        return

    configure_plot_style()
    fig_height = max(4.4, 0.58 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_height), dpi=300)
    y_positions = range(len(plot_data))
    bar_height = 0.38

    ax.barh(
        [y - bar_height / 2 for y in y_positions],
        plot_data[province_names[0]],
        height=bar_height,
        label=f"{province_names[0]} (most negative)",
        color=SENTIMENT_COLORS["negative"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.barh(
        [y + bar_height / 2 for y in y_positions],
        plot_data[province_names[1]],
        height=bar_height,
        label=f"{province_names[1]} (most positive)",
        color=SENTIMENT_COLORS["positive"],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_data.index)
    ax.set_xlabel("Share of frame mentions within province")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    xmax = float(plot_data.max().max())
    ax.set_xlim(0, xmax * 1.2 if xmax else 1)
    offset = xmax * 0.015 if xmax else 0.02
    for idx, (_, row) in enumerate(plot_data.iterrows()):
        for y_pos, province_name in [
            (idx - bar_height / 2, province_names[0]),
            (idx + bar_height / 2, province_names[1]),
        ]:
            value = row[province_name]
            if pd.notna(value):
                ax.text(value + offset, y_pos, f"{value:.1f}%", va="center", ha="left", fontsize=9)

    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_region_frame_figures(
    admin_df: pd.DataFrame,
    province_tbl: pd.DataFrame,
    frame_order: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    region_summary = eligible_province_summary(province_tbl)
    region_counts = build_region_frame_counts(admin_df, province_tbl)

    region_counts.to_csv(out_dir / "province_frame_counts.csv", index=False)
    region_summary.to_csv(out_dir / "province_sentiment_summary.csv", index=False)

    for region_name, region_df in region_counts.groupby("province_name", sort=True):
        plot_single_region_frame_counts(
            region_df,
            str(region_name),
            out_dir / f"{slugify(str(region_name))}_frames.png",
        )

    plot_region_sentiment_extreme_frame_comparison(
        region_counts,
        region_summary,
        frame_order,
        out_dir / "most_negative_vs_most_positive_province_frames.png",
    )
    plot_region_sentiment_extreme_frame_relative_importance(
        region_counts,
        region_summary,
        frame_order,
        out_dir / "most_negative_vs_most_positive_province_frame_shares.png",
    )


def build_frame_keyword_sentiment_data(
    categories_df: pd.DataFrame,
    frame_keywords: dict[str, set[str]],
    display_lookup: dict[str, dict[str, str]],
) -> pd.DataFrame:
    category_col = "matched_categories_str"
    keyword_col = "matched_keywords_str"
    sentiment_col = "sentiment_norm" if "sentiment_norm" in categories_df.columns else "sentiment"
    required = [category_col, keyword_col, sentiment_col]
    missing = [c for c in required if c not in categories_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in categories CSV: {missing}")

    records: list[dict[str, str]] = []
    df = categories_df[[category_col, keyword_col, sentiment_col]].copy()
    df["_sent"] = normalize_sentiment(df[sentiment_col])
    df = df[df["_sent"].isin(SENTIMENT_ORDER)].copy()

    for categories_value, keywords_value, sentiment in df[[category_col, keyword_col, "_sent"]].itertuples(index=False, name=None):
        categories = parse_semicolon_values(categories_value)
        keywords = parse_semicolon_values(keywords_value)
        if not categories or not keywords:
            continue

        keyword_norms = {kw.lower() for kw in keywords}
        for frame in categories:
            valid_keywords = frame_keywords.get(frame)
            if not valid_keywords:
                continue
            matched = sorted(keyword_norms & valid_keywords)
            for keyword_norm in matched:
                records.append(
                    {
                        "frame": frame,
                        "keyword": display_lookup.get(frame, {}).get(keyword_norm, keyword_norm),
                        "sentiment": sentiment,
                    }
                )

    if not records:
        return pd.DataFrame(columns=["frame", "keyword", "sentiment"])
    return pd.DataFrame.from_records(records)


def plot_frame_keyword_sentiment_distribution(
    pairs_df: pd.DataFrame,
    frame_order: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for frame in frame_order:
        frame_path = out_dir / f"{slugify(frame)}.png"
        frame_df = pairs_df[pairs_df["frame"] == frame].copy()

        if frame_df.empty:
            configure_plot_style()
            fig, ax = plt.subplots(figsize=(8, 3), dpi=250)
            ax.text(0.5, 0.5, "No validated keyword matches for this frame.", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(frame)
            plt.tight_layout()
            plt.savefig(frame_path, bbox_inches="tight")
            plt.close(fig)
            summary_rows.append(
                {
                    "frame": frame,
                    "figure_path": frame_path.name,
                    "n_keyword_mentions": 0,
                    "n_keywords_plotted": 0,
                }
            )
            continue

        counts = (
            frame_df.groupby(["keyword", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SENTIMENT_ORDER, fill_value=0)
        )
        counts["total"] = counts.sum(axis=1)
        counts = counts.sort_values(["total", "positive", "neutral", "negative"], ascending=[False, False, False, False])
        top_counts = counts.copy().sort_values("total", ascending=True)
        denom = top_counts["total"].replace({0: pd.NA})
        plot_data = top_counts[SENTIMENT_ORDER].div(denom, axis=0) * 100

        configure_plot_style()
        fig_height = max(3.6, 1.15 * len(top_counts) + 1.6)
        fig, ax = plt.subplots(figsize=(9.5, fig_height), dpi=250)
        left = pd.Series(0, index=plot_data.index, dtype=float)

        for sentiment in SENTIMENT_ORDER:
            ax.barh(
                plot_data.index,
                plot_data[sentiment],
                left=left,
                label=sentiment.capitalize(),
                color=SENTIMENT_COLORS[sentiment],
                edgecolor="white",
                linewidth=0.7,
            )
            left += plot_data[sentiment]

        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of keyword mentions")
        ax.set_title(f"{frame}: top 5 keywords by sentiment")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")
        ax.set_axisbelow(True)

        for keyword, total in top_counts["total"].items():
            ax.text(
                101,
                keyword,
                f"n={int(total)}",
                va="center",
                ha="left",
                fontsize=9,
                color="#3a3a3a",
            )

        ax.legend(
            title="Sentiment",
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
        )

        plt.tight_layout()
        plt.savefig(frame_path, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append(
            {
                "frame": frame,
                "figure_path": frame_path.name,
                "n_keyword_mentions": int(frame_df.shape[0]),
                "n_keywords_plotted": int(top_counts.shape[0]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "frame_keyword_figure_index.csv", index=False)


def truncate_text(text: str, limit: int = 220) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def write_empty_locations_html(out_path: Path, message: str) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interactive matched locations</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Georgia, "Times New Roman", serif;
      background: #f6f1e8;
      color: #3d342b;
      padding: 24px;
      box-sizing: border-box;
    }}
    .panel {{
      max-width: 720px;
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(83,72,61,0.18);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(61,52,43,0.08);
      padding: 24px 28px;
    }}
    h1 {{
      margin: 0 0 12px 0;
      font-size: 1.5rem;
      font-weight: 600;
    }}
    p {{
      margin: 0;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Interactive matched locations</h1>
    <p>{escape(message)}</p>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def plot_locations_interactive(
    admin_df: pd.DataFrame,
    province_gpkg: Path,
    out_path: Path,
) -> None:
    required = {"lon", "lat"}
    missing = required - set(admin_df.columns)
    if missing:
        write_empty_locations_html(
            out_path,
            f"Location map unavailable because the administrative CSV is missing coordinate columns: {sorted(missing)}.",
        )
        return

    points_df = admin_df.copy()
    points_df["lon"] = pd.to_numeric(points_df["lon"], errors="coerce")
    points_df["lat"] = pd.to_numeric(points_df["lat"], errors="coerce")
    points_df = points_df.dropna(subset=["lon", "lat"]).copy()
    if points_df.empty:
        write_empty_locations_html(
            out_path,
            "No valid lon/lat rows were available for the interactive location map. Upstream geocoding produced no mappable sentence locations for this run.",
        )
        return

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

    prov_xmin, prov_ymin, prov_xmax, prov_ymax = provinces.total_bounds
    xmin = min(prov_xmin, float(points_df["lon"].min()))
    xmax = max(prov_xmax, float(points_df["lon"].max()))
    ymin = min(prov_ymin, float(points_df["lat"].min()))
    ymax = max(prov_ymax, float(points_df["lat"].max()))
    xpad = max((xmax - xmin) * 0.08, 0.5)
    ypad = max((ymax - ymin) * 0.08, 0.5)

    fig.update_geos(
        fitbounds=False,
        showcountries=True,
        countrycolor="#b9b9b9",
        countrywidth=0.7,
        showcoastlines=True,
        coastlinecolor="#9f9f9f",
        coastlinewidth=0.7,
        showland=True,
        landcolor="#f7f2ea",
        showocean=True,
        oceancolor="#dbeaf2",
        showlakes=False,
        showrivers=False,
        bgcolor="#f6f1e8",
        lonaxis_range=[xmin - xpad, xmax + xpad],
        lataxis_range=[ymin - ypad, ymax + ypad],
        projection_type="natural earth",
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
    frame_order, frame_keywords, display_lookup = load_frame_keyword_vocab(Path(args.keywords_csv))
    province_gpkg = Path(args.province_gpkg)
    location_province_overrides = load_location_province_overrides(
        Path(args.location_province_overrides) if args.location_province_overrides else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_keywords_dir = output_dir / "frame_keywords"
    region_frames_dir = output_dir / "region_frames"

    province_tbl = build_province_summary(admin_df, args.country, location_province_overrides)
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
    plot_region_frame_figures(
        admin_df,
        province_tbl,
        frame_order,
        region_frames_dir,
    )
    plot_locations_interactive(
        admin_df,
        province_gpkg,
        output_dir / "locations_map.html",
    )
    frame_keyword_pairs = build_frame_keyword_sentiment_data(
        categories_df,
        frame_keywords,
        display_lookup,
    )
    plot_frame_keyword_sentiment_distribution(
        frame_keyword_pairs,
        frame_order,
        frame_keywords_dir,
    )

    print("Wrote:", output_dir / "province_sentiment_table.csv")
    print("Wrote:", output_dir / "provinces_sentiment_balance.png")
    print("Wrote:", output_dir / "provinces_sentiment_distribution.png")
    print("Wrote:", output_dir / "categories_sentiment_distribution.png")
    print("Wrote:", output_dir / "locations_map.html")
    print("Wrote:", frame_keywords_dir)
    print("Wrote:", region_frames_dir)


if __name__ == "__main__":
    main()
