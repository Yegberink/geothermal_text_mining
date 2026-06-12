#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import pandas as pd
from shapely.geometry import LineString, Point, box

from shape_resources import load_shapes_parquet, nuts2_shapes

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
MIN_PROVINCE_SENTENCES = 100
STACKED_FRAME_ORDER = [
    "Operational risk",
    "Technological uncertainty",
    "Public acceptance",
    "Permitting & policy",
    "Costs",
    "Infrastructure",
    "Sustainability",
]
STACKED_FRAME_COLORS = {
    "Operational risk": "#4C78A8",
    "Technological uncertainty": "#F58518",
    "Public acceptance": "#54A24B",
    "Permitting & policy": "#B279A2",
    "Costs": "#9D755D",
    "Infrastructure": "#72B7B2",
    "Sustainability": "#E45756",
}


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
    ap.add_argument("--shapes-parquet", type=str, default="data/shapes.parquet")
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

    fig_height = max(6.4, 0.38 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(10.8, fig_height), dpi=300)
    y_positions = list(range(len(plot_df)))

    for idx, row in plot_df.iterrows():
        color = row["country_color"]
        ax.barh(idx, -row["pct_neg"], color=color, alpha=0.34, edgecolor="white", linewidth=0.7)
        ax.barh(idx, row["pct_pos"], color=color, alpha=0.92, edgecolor="white", linewidth=0.7)

    bar_extent = float(plot_df[["pct_neg", "pct_pos"]].abs().max().max())
    balance_extent = float(plot_df["polarity_balance"].abs().max())
    base_extent = max(60, bar_extent, balance_extent)
    balance_label_offset = max(3.0, base_extent * 0.04)
    label_cushion = max(8.0, base_extent * 0.1)
    x_limit = max(60, bar_extent + label_cushion, balance_extent + balance_label_offset + label_cushion)
    ax.scatter(
        plot_df["polarity_balance"],
        y_positions,
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
            balance + (balance_label_offset if balance >= 0 else -balance_label_offset),
            idx,
            f"{balance:+.0f}%",
            ha="left" if balance >= 0 else "right",
            va="center",
            fontsize=8,
            color="#1f1f1f",
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="none", alpha=0.82),
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage of sentences (%)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):.0f}%"))
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.75, len(plot_df) - 0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["display_name"])
    for tick_label, is_average in zip(ax.get_yticklabels(), plot_df["is_country_average"], strict=False):
        if bool(is_average):
            tick_label.set_fontweight("bold")
    ax.invert_yaxis()

    n_label_x = x_limit + max(1.5, x_limit * 0.025)
    for idx, total in enumerate(plot_df["n_text_units"]):
        ax.text(
            n_label_x,
            idx,
            f"n={int(total)}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#3a3a3a",
            clip_on=False,
        )

    for _, group in plot_df.groupby("country", sort=False):
        start = int(group.index.min())
        if start > 0:
            ax.axhline(start - 0.5, color="#cfcfcf", linewidth=0.8)

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
        bbox_to_anchor=(0.5, 1.04),
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.grid(False)

    fig.tight_layout(rect=(0, 0, 0.96, 0.92))
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

    frame_df = admin_df[["language", "country", "country_color", "province_name", "matched_categories_str", "_sent"]].copy()
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
    ax.set_xlabel("Sentiment balance")
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


def most_negative_positive_provinces(province_summary: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    province_summary = province_summary[~province_summary["is_country_average"]].copy()
    if province_summary.empty or province_summary["province_name"].nunique() < 2:
        return None

    negative_region = province_summary.sort_values(
        ["polarity_balance", "n_text_units", "province_name"],
        ascending=[True, False, True],
    ).iloc[0]
    positive_candidates = province_summary[
        province_summary["province_name"].astype(str).ne(str(negative_region["province_name"]))
    ]
    if positive_candidates.empty:
        return None

    positive_region = positive_candidates.sort_values(
        ["polarity_balance", "n_text_units", "province_name"],
        ascending=[False, False, True],
    ).iloc[0]
    return negative_region, positive_region


def build_country_extreme_province_frame_share_table(
    admin_df: pd.DataFrame,
    province_tbl: pd.DataFrame,
    languages: list[str],
) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []
    for language in languages:
        language_summary = province_tbl[province_tbl["language"].astype(str).eq(str(language))]
        pair = most_negative_positive_provinces(language_summary)
        if pair is None:
            continue
        for role, row in [("Most negative province", pair[0]), ("Most positive province", pair[1])]:
            selected_rows.append(
                {
                    "language": row["language"],
                    "country": row["country"],
                    "country_color": row["country_color"],
                    "province_role": role,
                    "province_name": row["province_name"],
                    "province_polarity_balance": row["polarity_balance"],
                    "province_n_text_units": row["n_text_units"],
                }
            )

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "language",
                "country",
                "country_color",
                "province_role",
                "province_name",
                "province_polarity_balance",
                "province_n_text_units",
                "frame",
                "n_mentions",
                "n_total_frame_mentions",
                "share_pct",
            ]
        )

    frame_df = explode_frame_sentiments(admin_df)
    frame_df["province_name"] = frame_df.get("province_name", pd.Series(index=frame_df.index, dtype=object)).astype(str).str.strip()
    selected = selected.copy()
    selected["province_name"] = selected["province_name"].astype(str).str.strip()

    selected_frame_df = frame_df.merge(
        selected[
            [
                "language",
                "country",
                "country_color",
                "province_role",
                "province_name",
                "province_polarity_balance",
                "province_n_text_units",
            ]
        ],
        on=["language", "country", "country_color", "province_name"],
        how="inner",
    )
    if selected_frame_df.empty:
        return selected.assign(frame=pd.NA, n_mentions=0, n_total_frame_mentions=0, share_pct=0.0)

    counts = (
        selected_frame_df.groupby(
            [
                "language",
                "country",
                "country_color",
                "province_role",
                "province_name",
                "province_polarity_balance",
                "province_n_text_units",
                "frame",
            ]
        )
        .size()
        .rename("n_mentions")
        .reset_index()
    )
    totals = (
        counts.groupby(["language", "country", "province_role", "province_name"])["n_mentions"]
        .sum()
        .rename("n_total_frame_mentions")
        .reset_index()
    )
    counts = counts.merge(totals, on=["language", "country", "province_role", "province_name"], how="left")
    counts["share_pct"] = 100 * counts["n_mentions"] / counts["n_total_frame_mentions"].replace({0: pd.NA})
    counts["share_pct"] = counts["share_pct"].fillna(0)

    all_frames = sorted(counts["frame"].dropna().astype(str).unique())
    full_index = selected.merge(pd.DataFrame({"frame": all_frames}), how="cross")
    out = full_index.merge(
        counts,
        on=[
            "language",
            "country",
            "country_color",
            "province_role",
            "province_name",
            "province_polarity_balance",
            "province_n_text_units",
            "frame",
        ],
        how="left",
    )
    out["n_mentions"] = out["n_mentions"].fillna(0).astype(int)
    out["n_total_frame_mentions"] = out["n_total_frame_mentions"].fillna(0).astype(int)
    out["share_pct"] = out["share_pct"].fillna(0)

    frame_totals = out.groupby("frame")["n_mentions"].sum().sort_values(ascending=True)
    frame_order = {frame: idx for idx, frame in enumerate(frame_totals.index)}
    role_order = {"Most negative province": 0, "Most positive province": 1}
    language_order = {language: idx for idx, language in enumerate(languages)}
    out["_frame_order"] = out["frame"].map(frame_order).fillna(len(frame_order)).astype(int)
    out["_role_order"] = out["province_role"].map(role_order).fillna(len(role_order)).astype(int)
    out["_language_order"] = out["language"].map(language_order).fillna(len(language_order)).astype(int)
    out = out.sort_values(["_language_order", "_frame_order", "_role_order"]).reset_index(drop=True)
    return out.drop(columns=["_language_order", "_frame_order", "_role_order"])


def prepare_stacked_frame_data(extreme_tbl: pd.DataFrame) -> pd.DataFrame:
    if extreme_tbl.empty:
        return pd.DataFrame(columns=["country_group", "region", "sentiment_type", "frame", "percentage"])

    role_lookup = {
        "Most negative province": "Most negative",
        "Most positive province": "Most positive",
    }
    tidy = extreme_tbl[
        ["country", "province_name", "province_role", "frame", "share_pct"]
    ].rename(
        columns={
            "country": "country_group",
            "province_name": "region",
            "share_pct": "percentage",
        }
    )
    tidy["sentiment_type"] = tidy["province_role"].map(role_lookup).fillna(tidy["province_role"])
    tidy = tidy.drop(columns=["province_role"])

    selected = tidy[["country_group", "region", "sentiment_type"]].drop_duplicates()
    full_index = selected.merge(pd.DataFrame({"frame": STACKED_FRAME_ORDER}), how="cross")
    tidy = full_index.merge(
        tidy,
        on=["country_group", "region", "sentiment_type", "frame"],
        how="left",
    )
    tidy["percentage"] = tidy["percentage"].fillna(0.0)

    country_order = {
        country: idx
        for idx, country in enumerate(extreme_tbl["country"].drop_duplicates().astype(str))
    }
    sentiment_order = {"Most negative": 0, "Most positive": 1}
    frame_order = {frame: idx for idx, frame in enumerate(STACKED_FRAME_ORDER)}
    tidy["_country_order"] = tidy["country_group"].map(country_order).fillna(len(country_order)).astype(int)
    tidy["_sentiment_order"] = tidy["sentiment_type"].map(sentiment_order).fillna(len(sentiment_order)).astype(int)
    tidy["_frame_order"] = tidy["frame"].map(frame_order).fillna(len(frame_order)).astype(int)
    tidy = tidy.sort_values(["_country_order", "_sentiment_order", "_frame_order"]).reset_index(drop=True)
    return tidy.drop(columns=["_country_order", "_sentiment_order", "_frame_order"])


def contrast_text_color(hex_color: str) -> str:
    r, g, b, _ = to_rgba(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "#1f1f1f"


def plot_frame_composition_100pct(stacked_df: pd.DataFrame, png_path: Path, pdf_path: Path | None = None) -> None:
    if stacked_df.empty:
        raise ValueError("No stacked frame composition rows were available to plot.")

    row_meta = stacked_df[["country_group", "region", "sentiment_type"]].drop_duplicates().reset_index(drop=True)
    y_positions: list[float] = []
    y_labels: list[str] = []

    current_y = 0.0
    for _, country_rows in row_meta.groupby("country_group", sort=False):
        for _, row in country_rows.iterrows():
            y_positions.append(current_y)
            y_labels.append(f"{row['region']}\n{row['sentiment_type']}")
            current_y += 1.0
        current_y += 0.58

    row_meta = row_meta.copy()
    row_meta["_y"] = y_positions
    plot_df = stacked_df.merge(row_meta, on=["country_group", "region", "sentiment_type"], how="left")

    totals = (
        plot_df.groupby(["country_group", "region", "sentiment_type"])["percentage"]
        .sum()
        .rename("_total")
        .reset_index()
    )
    plot_df = plot_df.merge(totals, on=["country_group", "region", "sentiment_type"], how="left")
    plot_df["_plot_percentage"] = 100 * plot_df["percentage"] / plot_df["_total"].replace({0: pd.NA})
    plot_df["_plot_percentage"] = plot_df["_plot_percentage"].fillna(0)

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=300)
    bar_height = 0.68

    for _, row in row_meta.iterrows():
        row_values = (
            plot_df[
                plot_df["country_group"].eq(row["country_group"])
                & plot_df["region"].eq(row["region"])
                & plot_df["sentiment_type"].eq(row["sentiment_type"])
            ]
            .set_index("frame")
            .reindex(STACKED_FRAME_ORDER)
        )
        left = 0.0
        for frame in STACKED_FRAME_ORDER:
            value = float(row_values.loc[frame, "_plot_percentage"])
            label_value = float(row_values.loc[frame, "percentage"])
            color = STACKED_FRAME_COLORS[frame]
            if value <= 0:
                continue
            ax.barh(
                row["_y"],
                value,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )
            if label_value >= 8 and value >= 6:
                ax.text(
                    left + value / 2,
                    row["_y"],
                    f"{label_value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=contrast_text_color(color),
                )
            left += value

    ax.set_xlim(0, 100)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_ylim(max(y_positions) + 0.75, -0.9)
    ax.set_xlabel("Share of frame mentions")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(axis="y", length=0)
    ax.grid(False)

    legend_handles = [
        mpatches.Patch(facecolor=STACKED_FRAME_COLORS[frame], edgecolor="white", label=frame)
        for frame in STACKED_FRAME_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=1.2,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.08)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_province_polygons(
    languages: list[str],
    countries: list[str],
    shapes_parquet: str | Path,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    shapes = load_shapes_parquet(shapes_parquet)
    for language, country in zip(languages, countries, strict=True):
        gdf = nuts2_shapes(shapes, country)
        gdf = gdf[["parent_id", "parent_name", "geometry"]].rename(
            columns={"parent_id": "province_code", "parent_name": "province_name"}
        ).copy()
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


def load_country_outline_polygons(
    _countries: list[str],
    shapes_parquet: str | Path,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    shapes = load_shapes_parquet(shapes_parquet)
    context = shapes[
        shapes["shape_class"].astype(str).str.lower().eq("land")
        & shapes["parent"].astype(str).str.lower().eq("nuts")
        & shapes["parent_subtype"].astype(str).eq("0")
    ].copy()
    if not context.empty:
        context = context[["country_id", "parent_name", "geometry"]].rename(columns={"parent_name": "country_name"})
        context["is_focus_country"] = False
        frames.append(context.to_crs("EPSG:4326"))

    if not frames:
        return gpd.GeoDataFrame(
            columns=["country_id", "country_name", "is_focus_country", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    outlines = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True, sort=False), geometry="geometry", crs="EPSG:4326")
    outlines = outlines.sort_values(["country_id", "is_focus_country"]).drop_duplicates(subset=["country_id"], keep="last")
    outlines = outlines.reset_index(drop=True)
    outlines["geometry"] = outlines.geometry.simplify(0.01, preserve_topology=True)
    outlines["_country_feature_id"] = outlines.index.astype(str)
    return outlines


def build_map_polygons(
    admin_df: pd.DataFrame,
    province_polygons: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    df = admin_df.copy()
    has_province = df["province_name"].notna() & df["province_name"].astype(str).str.strip().ne("")
    province_df = df.loc[has_province].copy()
    province_df["province_name"] = province_df["province_name"].astype(str).str.strip()
    province_df["_province_key"] = province_df["province_name"].map(normalize_key)

    province_counts = (
        province_df.groupby(["language", "country", "_province_key", "_sent"])
        .size()
        .unstack(fill_value=0)
    )
    province_scores = counts_to_summary(province_counts).reset_index()
    province_scores["has_enough_data"] = province_scores["n_text_units"] >= MIN_PROVINCE_SENTENCES

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
                "has_enough_data",
            ]
        ],
        on=["language", "country", "_province_key"],
        how="left",
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged["n_text_units"] = merged["n_text_units"].fillna(0).astype(int)
    merged["has_enough_data"] = merged["has_enough_data"].where(merged["has_enough_data"].notna(), False).astype(bool)
    merged["geometry"] = merged.geometry.simplify(0.01, preserve_topology=True)
    merged = merged.reset_index(drop=True)
    merged["_feature_id"] = merged.index.astype(str)
    return merged


def build_map_points(admin_df: pd.DataFrame) -> pd.DataFrame:
    if not {"lon", "lat"}.issubset(admin_df.columns):
        return pd.DataFrame(columns=["lon", "lat", "sentiment"])
    points = admin_df[["lon", "lat", "_sent", "language", "country", "province_name"]].copy()
    points["lon"] = pd.to_numeric(points["lon"], errors="coerce")
    points["lat"] = pd.to_numeric(points["lat"], errors="coerce")
    return points.dropna(subset=["lon", "lat"]).copy()


def plot_static_sentiment_map(
    polygons: gpd.GeoDataFrame,
    points: pd.DataFrame,
    country_outlines: gpd.GeoDataFrame,
    out_path: Path,
) -> None:
    if polygons.empty:
        raise ValueError("No province polygons matched the cross-language sentiment table.")

    map_crs = "EPSG:3035"
    lon_min, lon_max = -12, 35
    lat_min, lat_max = 35, 58
    extent = gpd.GeoSeries([box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326").to_crs(map_crs).total_bounds
    xmin, ymin, xmax, ymax = extent

    plot_polygons = polygons.to_crs(map_crs)
    outlines = country_outlines.to_crs(map_crs) if not country_outlines.empty else country_outlines
    context_outlines = outlines

    cmap = LinearSegmentedColormap.from_list("sentiment_balance", ["#8C510A", "#F5F5F5", "#01665E"])
    scored_polygons = plot_polygons[plot_polygons["has_enough_data"]].copy()
    low_data_polygons = plot_polygons[~plot_polygons["has_enough_data"]].copy()
    abs_balance = max(1.0, float(scored_polygons["polarity_balance"].abs().max()) if not scored_polygons.empty else 1.0)
    norm = TwoSlopeNorm(vmin=-abs_balance, vcenter=0, vmax=abs_balance)

    fig, ax = plt.subplots(figsize=(10.8, 8.4), dpi=220)
    ax.set_facecolor("#DDE9EC")

    if not context_outlines.empty:
        context_outlines.plot(ax=ax, facecolor="#EFE8DA", edgecolor="#B7AA94", linewidth=0.35, zorder=1)
    if not low_data_polygons.empty:
        low_data_polygons.plot(ax=ax, facecolor="#C9C9C9", edgecolor="white", linewidth=0.35, zorder=2)
    if not scored_polygons.empty:
        scored_polygons.plot(
            ax=ax,
            column="polarity_balance",
            cmap=cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    if not points.empty:
        point_gdf = gpd.GeoDataFrame(
            points.copy(),
            geometry=[Point(lon, lat) for lon, lat in zip(points["lon"], points["lat"], strict=False)],
            crs="EPSG:4326",
        ).to_crs(map_crs)
        ax.scatter(
            point_gdf.geometry.x,
            point_gdf.geometry.y,
            s=1.35,
            c="#202020",
            alpha=0.14,
            linewidths=0,
            zorder=4,
        )

    def _project_line(coords: list[tuple[float, float]]):
        return gpd.GeoSeries([LineString(coords)], crs="EPSG:4326").to_crs(map_crs).iloc[0]

    for lon in range(-10, 31, 10):
        line = _project_line([(lon, lat_min + (lat_max - lat_min) * i / 80) for i in range(81)])
        ax.plot(*line.xy, color="#252525", alpha=0.16, linewidth=0.45, zorder=0)
    for lat in range(35, 56, 5):
        line = _project_line([(lon_min + (lon_max - lon_min) * i / 100, lat) for i in range(101)])
        ax.plot(*line.xy, color="#252525", alpha=0.16, linewidth=0.45, zorder=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    def _format_lon(value: int) -> str:
        if value < 0:
            return f"{abs(value)}W"
        if value > 0:
            return f"{value}E"
        return "0"

    for lon in range(-10, 31, 10):
        label_point = gpd.GeoSeries([Point(lon, lat_min)], crs="EPSG:4326").to_crs(map_crs).iloc[0]
        ax.text(label_point.x, ymin - 72000, _format_lon(lon), ha="center", va="top", fontsize=7.5, color="#555047")
    for lat in range(35, 56, 5):
        label_point = gpd.GeoSeries([Point(lon_min, lat)], crs="EPSG:4326").to_crs(map_crs).iloc[0]
        ax.text(xmin - 76000, label_point.y, f"{lat}N", ha="right", va="center", fontsize=7.5, color="#555047")

    scale_km = 500
    scale_m = scale_km * 1000
    scale_x0 = xmin + 0.065 * (xmax - xmin)
    scale_y = ymin + 0.075 * (ymax - ymin)
    tick_h = 36000
    ax.plot([scale_x0, scale_x0 + scale_m], [scale_y, scale_y], color="#252525", linewidth=2.0, zorder=6)
    ax.plot([scale_x0, scale_x0], [scale_y - tick_h / 2, scale_y + tick_h / 2], color="#252525", linewidth=1.2, zorder=6)
    ax.plot(
        [scale_x0 + scale_m, scale_x0 + scale_m],
        [scale_y - tick_h / 2, scale_y + tick_h / 2],
        color="#252525",
        linewidth=1.2,
        zorder=6,
    )
    ax.text(scale_x0 + scale_m / 2, scale_y + 52000, f"{scale_km} km", ha="center", va="bottom", fontsize=8.5, color="#252525")

    ax.annotate(
        "",
        xy=(0.08, 0.91),
        xytext=(0.08, 0.81),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#252525", linewidth=1.7, mutation_scale=18),
        zorder=7,
    )
    ax.text(
        0.08,
        0.935,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#252525",
        zorder=7,
        bbox=dict(boxstyle="square,pad=0.18", facecolor="#F3EFE7", edgecolor="#252525", linewidth=0.6, alpha=0.86),
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#252525")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.034, pad=0.025)
    cbar.set_label("Sentiment balance (positive - negative, pp)")
    cbar.outline.set_linewidth(0.6)

    legend_handles = [
        mpatches.Patch(facecolor="#EFE8DA", edgecolor="#B7AA94", label="Country background"),
        mpatches.Patch(facecolor="#C9C9C9", edgecolor="white", label=f"<{MIN_PROVINCE_SENTENCES} sentences"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#202020", alpha=0.35, markersize=4, label="Sentence location"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True,
        framealpha=0.82,
        facecolor="#F3EFE7",
        edgecolor="#B7AA94",
    )
    ax.text(
        0.995,
        -0.055,
        "Projection: ETRS89 / LAEA Europe (EPSG:3035). Boundaries: data/shapes.parquet.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555047",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    lengths = {
        "languages": len(args.languages),
        "countries": len(args.countries),
        "admin_csvs": len(args.admin_csvs),
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

    extreme_frame_share_tbl = build_country_extreme_province_frame_share_table(admin_df, province_tbl, args.languages)
    extreme_frame_share_tbl.to_csv(output_dir / "all_languages_extreme_province_frame_shares_table.csv", index=False)
    stacked_frame_tbl = prepare_stacked_frame_data(extreme_frame_share_tbl)
    stacked_frame_tbl.to_csv(output_dir / "frame_mentions_100pct_stacked_table.csv", index=False)
    plot_frame_composition_100pct(
        stacked_frame_tbl,
        output_dir / "frame_mentions_100pct_stacked.png",
        output_dir / "frame_mentions_100pct_stacked.pdf",
    )

    province_polygons = load_province_polygons(args.languages, args.countries, args.shapes_parquet)
    map_polygons = build_map_polygons(admin_df, province_polygons)
    map_points = build_map_points(admin_df)
    country_outlines = load_country_outline_polygons(args.countries, args.shapes_parquet)
    plot_static_sentiment_map(
        map_polygons,
        map_points,
        country_outlines,
        output_dir / "all_languages_province_sentiment_map.png",
    )

    print("Wrote:", output_dir / "all_languages_province_sentiment_table.csv")
    print("Wrote:", output_dir / "all_languages_province_sentiment_balance.png")
    print("Wrote:", output_dir / "all_languages_frames_sentiment_table.csv")
    print("Wrote:", output_dir / "all_languages_frames_sentiment_distribution.png")
    print("Wrote:", output_dir / "all_languages_frames_country_sentiment_balance_table.csv")
    print("Wrote:", output_dir / "all_languages_frames_country_sentiment_balance.png")
    print("Wrote:", output_dir / "all_languages_extreme_province_frame_shares_table.csv")
    print("Wrote:", output_dir / "frame_mentions_100pct_stacked_table.csv")
    print("Wrote:", output_dir / "frame_mentions_100pct_stacked.png")
    print("Wrote:", output_dir / "frame_mentions_100pct_stacked.pdf")
    print("Wrote:", output_dir / "all_languages_province_sentiment_map.png")


if __name__ == "__main__":
    main()
