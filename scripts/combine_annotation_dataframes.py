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


def language_from_path(path: Path) -> str:
    parts = path.parts
    if "annotation" in parts:
        idx = parts.index("annotation")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def main() -> None:
    args = parse_args()
    frames = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        df = pd.read_csv(input_path)
        if "language" not in df.columns:
            df.insert(0, "language", language_from_path(input_path))
        frames.append(df)

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote: {output_path} (rows={len(out)})")


if __name__ == "__main__":
    main()
