#!/usr/bin/env python3
"""Populate or refresh the local asset cache (`data/`) without running a notebook.

    python tools/fetch-assets.py <slug>                 # download what is missing
    python tools/fetch-assets.py <slug> --force         # delete the cache, re-fetch
    python tools/fetch-assets.py <slug> --from-mirror   # copy from mirror/, no network
    python tools/fetch-assets.py --all

WHY THIS EXISTS

`data/` was only ever filled as a SIDE EFFECT of `azimuth.setup()` inside a
notebook. So the only way to refresh a corrected dataset was to run the whole
workshop — and if the corrected file had not reached the URL the manifest
points at, running it just re-cached the stale copy and reported it verified.
There was no way to say "get me the current bytes and check them" on its own.

It uses the SHIM'S OWN fetcher, not a second download path. A refresh that
verified differently from a workshop run would be worse than no refresh: it
would tell you the file is fine in one place and fail in the other.

`--from-mirror` is the offline route, and the fast one for assets we host
ourselves: the file is already on disk under mirror/, so copying it and
checking the hash proves the pin is correct BEFORE anything is pushed. That is
the check to run after re-encoding a corpus and before `pin-asset --force`.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSHOPS = ROOT / "workshops"
MIRROR = ROOT / "mirror"
DATA = Path(__file__).resolve().parents[1] / "data"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirror_candidate(asset: dict) -> Path | None:
    """The local file behind a mirror URL, if this asset has one.

    The manifest's `name` is what the notebook opens; the mirror's filename is
    whatever the URL ends in. They are allowed to differ — parallel.tsv is
    served as ara-eng-parallel.tsv — so resolve through the URL rather than
    assuming they match.
    """
    for url in asset.get("sources") or []:
        if "azimuth-workshops" in url and "/mirror/" in url:
            candidate = MIRROR / url.rsplit("/", 1)[-1]
            if candidate.exists():
                return candidate
    direct = MIRROR / asset["name"]
    return direct if direct.exists() else None


def handle(slug: str, force: bool, from_mirror: bool) -> int:
    spec_path = WORKSHOPS / slug / "workshop.yaml"
    if not spec_path.exists():
        print(f"no such workshop: {slug}")
        return 1
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assets = spec.get("assets") or []
    if not assets:
        print(f"  {slug}: declares no assets")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    problems = 0

    for asset in assets:
        name = asset["name"]
        dest = DATA / name
        expected = asset.get("sha256")

        if force and dest.exists():
            dest.unlink()
            print(f"  · {name} — cached copy deleted")

        if from_mirror:
            source = mirror_candidate(asset)
            if source is None:
                print(f"  x {name} — no local file under mirror/ for this asset")
                problems += 1
                continue
            shutil.copyfile(source, dest)
            actual = sha256_of(dest)
            size = dest.stat().st_size
            if expected and actual != expected:
                # The whole point of --from-mirror: catch this BEFORE pushing,
                # while the fix is still one `pin-asset --force` away.
                print(f"  x {name} — mirror/{source.name} does NOT match the pinned hash")
                print(f"      pinned : {expected}")
                print(f"      on disk: {actual}  ({size:,} bytes)")
                print(
                    "      the manifest is stale -> tools/pin-asset.py "
                    f"{slug} --force (after pushing the file)"
                )
                problems += 1
            else:
                state = "matches the pin" if expected else "UNPINNED"
                print(f"  · {name} <- mirror/{source.name} — {size:,} bytes, {state}")
            continue

        # Network path: the shim's fetcher, so verification is identical to a
        # real workshop run (including AZ-E301/AZ-E302).
        sys.path.insert(0, str(ROOT / "shim"))
        from azimuth_nb.assets import fetch
        from azimuth_nb.errors import AzimuthError

        try:
            path = fetch(asset, DATA)
            print(f"  · {name} — {path.stat().st_size:,} bytes at {path}")
        except AzimuthError as exc:
            print(f"  x {exc}")
            problems += 1

    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="delete cached copies first")
    parser.add_argument(
        "--from-mirror",
        action="store_true",
        help="copy from the local mirror/ directory and verify — no network",
    )
    args = parser.parse_args()

    if args.all:
        slugs = sorted(d.name for d in WORKSHOPS.iterdir() if d.is_dir())
    elif args.slug:
        slugs = [args.slug]
    else:
        print(__doc__)
        return 2

    worst = 0
    for slug in slugs:
        print(f"{slug}:")
        worst = max(worst, handle(slug, args.force, args.from_mirror))
    return worst


if __name__ == "__main__":
    sys.exit(main())
