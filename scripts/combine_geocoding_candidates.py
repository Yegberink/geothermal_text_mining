#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    return ap.parse_args()


def parse_input_spec(spec: str) -> tuple[str, str, str, Path]:
    parts = spec.split("=", 3)
    if len(parts) != 4:
        raise ValueError(
            "Each --inputs value must be language=country=country_codes=path, "
            f"got {spec!r}"
        )
    language, country, country_codes, path = parts
    return language, country, country_codes, Path(path)


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for spec in args.inputs:
        language, country, country_codes, path = parse_input_spec(spec)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype=str).fillna("")
        if df.empty:
            continue
        df["language"] = language
        df["country"] = country
        df["country_codes"] = country_codes
        frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        if "normalized_location" in out.columns:
            out = out.drop_duplicates(["normalized_location", "country_codes"], keep="first")
        elif "location" in out.columns:
            out = out.drop_duplicates(["location", "country_codes"], keep="first")
        if "row_count" in out.columns:
            out["row_count_num"] = pd.to_numeric(out["row_count"], errors="coerce").fillna(0).astype(int)
            out = out.sort_values(["row_count_num", "language", "location"], ascending=[False, True, True])
            out = out.drop(columns=["row_count_num"])
    else:
        out = pd.DataFrame(
            columns=[
                "location",
                "normalized_location",
                "granularity",
                "row_count",
                "location_source",
                "examples",
                "language",
                "country",
                "country_codes",
            ]
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote combined geocoding candidates: {output} (rows={len(out)})")


if __name__ == "__main__":
    main()
