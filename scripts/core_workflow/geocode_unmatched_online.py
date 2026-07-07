#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
import urllib.request

from geopy.exc import GeocoderRateLimited, GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

from geocoding_online import bias_query, configure_ssl_cert_bundle, geocoder_cache_key

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
NOMINATIM_POLICY_URL = "https://operations.osmfoundation.org/policies/nominatim/"

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


class FatalGeocoderConfigurationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=str, default=str(DEFAULT_PROJECT_DIR))
    ap.add_argument("--input-unmatched-csv", type=str, default="output/generic/workflow/geocoding_online_candidates_all.csv")
    ap.add_argument("--output-cache", type=str, default="cache/geocode_unmatched_online.jsonl")
    ap.add_argument("--provider", choices=["none", "nominatim", "opencage", "photon"], default="none")
    ap.add_argument("--country", type=str, default="")
    ap.add_argument("--country-codes", type=str, default="")
    ap.add_argument("--user-agent", type=str, default="absa-geo-mapper")
    ap.add_argument("--lock-path", type=str, default="cache/geocode_unmatched_online.lock")
    ap.add_argument("--api-key", type=str, default=os.environ.get("GEOCODER_API_KEY", ""))
    ap.add_argument("--min-delay-seconds", type=float, default=1.1)
    ap.add_argument("--timeout-seconds", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-wait-seconds", type=float, default=20.0)
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--nominatim-policy-ack", action="store_true")
    return ap.parse_args()


@contextmanager
def single_process_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        if fcntl is None:
            yield
            return
        print(f"Waiting for geocoder cache lock: {lock_path}")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        print(f"Acquired geocoder cache lock: {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            print(f"Released geocoder cache lock: {lock_path}")


def load_unmatched_locations(path: Path, max_queries: int = 0) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing unmatched geocoding report: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if str(row.get("location", "")).strip()]
    rows.sort(key=lambda row: int(row.get("row_count") or 0), reverse=True)
    return rows[:max_queries] if max_queries > 0 else rows


def load_existing_statuses(cache_path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not cache_path.exists():
        return statuses
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(entry.get("key") or "")
            if key:
                statuses[key] = str(entry.get("status") or "")
    return statuses


def row_country(row: dict[str, str], fallback: str) -> str:
    return str(row.get("country") or fallback or "")


def row_country_codes(row: dict[str, str], fallback: str) -> str:
    return str(row.get("country_codes") or fallback or "")


def append_cache_entry(cache_path: Path, entry: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_entry(
    query: str,
    biased_query: str,
    country_codes: str,
    provider: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "key": geocoder_cache_key(query, country_codes),
        "query": query,
        "biased_query": biased_query,
        "country_codes": country_codes,
        "provider": provider,
        "status": status,
        "result": result,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_json(url: str, user_agent: str, timeout_seconds: int) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_photon(query: str, args: argparse.Namespace) -> dict[str, Any] | None:
    url = "https://photon.komoot.io/api/?" + urlencode({"q": query, "limit": 1})
    payload = fetch_json(url, args.user_agent, args.timeout_seconds)
    features = payload.get("features") or []
    if not features:
        return None
    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"][:2]
    props = feature.get("properties") or {}
    parts = [props.get("name"), props.get("city"), props.get("state"), props.get("country")]
    display = ", ".join(str(part) for part in parts if part)
    return {"lat": float(lat), "lon": float(lon), "disp": display or query, "provider": "photon"}


def geocode_opencage(query: str, args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.api_key:
        raise RuntimeError("OpenCage provider requires --api-key or GEOCODER_API_KEY.")
    params = {"q": query, "key": args.api_key, "limit": 1, "no_annotations": 1}
    if args.country_codes:
        params["countrycode"] = args.country_codes
    url = "https://api.opencagedata.com/geocode/v1/json?" + urlencode(params)
    payload = fetch_json(url, args.user_agent, args.timeout_seconds)
    results = payload.get("results") or []
    if not results:
        return None
    result = results[0]
    geometry = result.get("geometry") or {}
    return {
        "lat": float(geometry["lat"]),
        "lon": float(geometry["lng"]),
        "disp": str(result.get("formatted") or query),
        "provider": "opencage",
    }


def geocode_nominatim(query: str, geolocator: Nominatim, args: argparse.Namespace) -> dict[str, Any] | None:
    loc = geolocator.geocode(
        query,
        exactly_one=True,
        addressdetails=False,
        country_codes=args.country_codes or None,
    )
    if not loc:
        return None
    return {"lat": float(loc.latitude), "lon": float(loc.longitude), "disp": str(loc), "provider": "nominatim"}


def query_provider(query: str, geolocator: Nominatim | None, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.provider == "nominatim":
        if geolocator is None:
            raise RuntimeError("Nominatim geolocator was not initialized.")
        return geocode_nominatim(query, geolocator, args)
    if args.provider == "opencage":
        return geocode_opencage(query, args)
    if args.provider == "photon":
        return geocode_photon(query, args)
    return None


def geocode_with_retries(query: str, geolocator: Nominatim | None, args: argparse.Namespace) -> tuple[str, dict[str, Any] | None, str | None]:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            result = query_provider(query, geolocator, args)
            return ("ok", result, None) if result else ("no_result", None, None)
        except GeocoderRateLimited as exc:
            last_error = exc
            delay = max(float(getattr(exc, "retry_after", 0) or 0), args.retry_wait_seconds)
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                raise FatalGeocoderConfigurationError(
                    "Geocoder SSL certificate verification failed. The active Python environment "
                    "does not trust the HTTPS certificate chain used for the geocoder request. "
                    "Fix the environment CA bundle before rerunning; this is not a missing location."
                ) from exc
            delay = args.retry_wait_seconds
        if attempt < args.max_retries:
            print(f"[warning] {args.provider} failed for {query!r} ({type(last_error).__name__}); sleeping {delay:.1f}s")
            time.sleep(delay)
    return "error", None, f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"


def main() -> None:
    args = parse_args()
    configure_ssl_cert_bundle()
    project_dir = Path(args.project_dir).expanduser().resolve()
    os.chdir(project_dir)

    cache_path = Path(args.output_cache)
    if args.provider == "none":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.touch(exist_ok=True)
        print("No online geocoder selected; wrote/kept empty cache file:", cache_path)
        return
    if args.provider == "nominatim" and not args.nominatim_policy_ack:
        raise SystemExit(
            "Nominatim cache filling requires --nominatim-policy-ack. "
            f"Review the public API policy first: {NOMINATIM_POLICY_URL}"
        )

    default_country_codes = args.country_codes
    rows = load_unmatched_locations(Path(args.input_unmatched_csv), args.max_queries)
    statuses = load_existing_statuses(cache_path)
    todo = []
    for row in rows:
        query = str(row.get("location") or "").strip()
        country_codes = row_country_codes(row, default_country_codes)
        key = geocoder_cache_key(query, country_codes)
        status = statuses.get(key)
        if status in {"ok", "no_result"} or (status == "error" and not args.retry_errors):
            continue
        todo.append(row)

    estimated_seconds = len(todo) * max(args.min_delay_seconds, 0)
    print(f"Provider: {args.provider}")
    print(f"Unmatched locations loaded: {len(rows)}")
    print(f"Queries remaining after cache skip: {len(todo)}")
    print(f"Minimum ETA from delay alone: {estimated_seconds / 60:.1f} min")
    if args.provider == "nominatim":
        print(f"Nominatim policy: {NOMINATIM_POLICY_URL}")

    geolocator = Nominatim(user_agent=args.user_agent, timeout=args.timeout_seconds) if args.provider == "nominatim" else None

    with single_process_lock(Path(args.lock_path)):
        start = time.time()
        last_request_at = 0.0
        try:
            for attempted, row in enumerate(todo, start=1):
                query = str(row.get("location") or "").strip()
                country = row_country(row, args.country)
                country_codes = row_country_codes(row, default_country_codes)
                args.country_codes = country_codes
                biased_query = bias_query(query, country)
                wait = args.min_delay_seconds - (time.time() - last_request_at)
                if wait > 0:
                    time.sleep(wait)
                status, result, error = geocode_with_retries(biased_query, geolocator, args)
                last_request_at = time.time()
                append_cache_entry(
                    cache_path,
                    make_entry(query, biased_query, country_codes, args.provider, status, result, error),
                )
                if attempted % args.print_every == 0 or attempted == len(todo):
                    elapsed = time.time() - start
                    rate = attempted / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - attempted) / rate if rate > 0 else 0
                    print(f"[{attempted}/{len(todo)}] status={status} ETA={eta / 60:.1f} min")
        except KeyboardInterrupt:
            print(f"\nInterrupted. Completed queries have already been appended to {cache_path}.")
            raise


if __name__ == "__main__":
    main()
