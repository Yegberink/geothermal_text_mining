from __future__ import annotations


SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#D55E00",
    "neutral": "#8A8A8A",
    "positive": "#009E73",
}
COUNTRY_COLORS = {
    "Germany": "#4C78A8",
    "Austria": "#B279A2",
    "Switzerland": "#54A24B",
    "Netherlands": "#F58518",
    "Italy": "#72B7B2",
}
FALLBACK_COUNTRY_COLORS = ["#4C78A8", "#B279A2", "#54A24B", "#F58518", "#72B7B2", "#999999"]


def country_color(country: object) -> str:
    value = str(country or "").strip()
    if value in COUNTRY_COLORS:
        return COUNTRY_COLORS[value]
    return FALLBACK_COUNTRY_COLORS[abs(hash(value)) % len(FALLBACK_COUNTRY_COLORS)]
