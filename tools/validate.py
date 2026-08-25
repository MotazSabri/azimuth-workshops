#!/usr/bin/env python3
"""Validate every workshop in the repository. CPU only, no torch, no network.

These are the checks I ran by hand while building the first workshop, written
down so they cannot quietly stop being true. This is NOT the Phase 2 builder —
it does not generate anything. It only refuses a repository that has drifted
out of the shape the notebooks depend on.

Run it with `python tools/validate.py`. Exit code 1 on any error.

Each check exists because a specific failure is otherwise invisible until a
learner hits it:

  * an orphaned region  — code that nothing runs, usually left behind when a
    prose edit removed the cell that referenced it
  * a dangling ref      — a cell that would generate an empty code block
  * a duplicate cell id — two cells writing to the same key in the captured
    outputs manifest, so one silently overwrites the other
  * a missing `ar`      — the Arabic notebook degrading into a partly English
    one, which is the failure mode this whole project exists to avoid
  * a default profile over the free-tier ceiling — the one constraint that
    never gets relaxed
  * committed outputs   — a notebook shipped with someone's run baked in
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSHOPS = ROOT / "workshops"
NOTEBOOKS = ROOT / "generated" / "notebooks"

# Colab's free tier, as of the last time it was checked. A default profile that
# needs more than this is a workshop most readers cannot run.
FREE_TIER_VRAM_GB = 16
FREE_TIER_RAM_GB = 12

REGION_RE = re.compile(r"^# --8<-- \[start:(\w+)\]$", re.M)

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def bilingual_gaps(node: object, path: str = "") -> list[str]:
    """Every mapping that has an `en` but no `ar` (or the reverse)."""
    gaps: list[str] = []
    if isinstance(node, dict):
        has_en, has_ar = "en" in node, "ar" in node
        if has_en != has_ar:
            gaps.append(f"{path or '<root>'} has {'en' if has_en else 'ar'} only")
        for key, value in node.items():
            gaps += bilingual_gaps(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            gaps += bilingual_gaps(value, f"{path}[{i}]")
    return gaps


def check_workshop(directory: Path) -> None:
    slug = directory.name
    where = f"workshops/{slug}"

    spec_path = directory / "workshop.yaml"
    code_path = directory / "code.py"
    if not spec_path.exists():
        err(f"{where}: no workshop.yaml")
        return
    if not code_path.exists():
        err(f"{where}: no code.py")
        return

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    code = code_path.read_text(encoding="utf-8")

    if spec.get("id") != slug:
        err(f"{where}: id is {spec.get('id')!r} but the directory is {slug!r}")

    # ── regions ↔ refs, both directions ────────────────────────────────────
    # Only markers at the start of a line count, so a comment DESCRIBING the
    # syntax cannot register as a region. That ambiguity was a real finding on
    # the first workshop.
    regions = set(REGION_RE.findall(code))
    body = spec.get("body") or []
    refs = {b["ref"] for b in body if b.get("ref")}

    for orphan in sorted(regions - refs):
        err(f"{where}: region '{orphan}' in code.py is never referenced by a cell")
    for dangling in sorted(refs - regions):
        err(f"{where}: cell ref '{dangling}' has no matching region in code.py")

    # ── cell ids ───────────────────────────────────────────────────────────
    ids = [b["id"] for b in body if b.get("type") in ("cell", "exercise")]
    for cell_id in {i for i in ids if ids.count(i) > 1}:
        err(f"{where}: duplicate cell id '{cell_id}' — captured outputs would collide")
    for block in body:
        if block.get("captures") and not block.get("id"):
            err(f"{where}: a block with captures has no id to join them on")

    # ── both languages ─────────────────────────────────────────────────────
    for gap in bilingual_gaps(spec):
        err(f"{where}: {gap}")

    # ── profiles ───────────────────────────────────────────────────────────
    profiles = spec.get("profiles") or []
    defaults = [p for p in profiles if p.get("default")]
    if len(defaults) != 1:
        err(f"{where}: exactly one profile must be marked default (found {len(defaults)})")
    else:
        needs = defaults[0].get("requires") or {}
        if float(needs.get("vramGb", 0) or 0) > FREE_TIER_VRAM_GB:
            err(
                f"{where}: default profile needs {needs['vramGb']} GB VRAM, "
                f"above the free-tier ceiling of {FREE_TIER_VRAM_GB} GB"
            )
        if float(needs.get("ramGb", 0) or 0) > FREE_TIER_RAM_GB:
            err(f"{where}: default profile needs more RAM than the free tier provides")
        if not defaults[0].get("scale"):
            err(f"{where}: default profile declares no scale — sizes would have no source")

    # ── papers ─────────────────────────────────────────────────────────────
    papers = spec.get("papers") or []
    if not any(p.get("role") == "core" for p in papers):
        warn(f"{where}: no paper marked core — nothing will invite readers to this workshop")
    for paper in papers:
        if paper.get("role") not in ("core", "supporting"):
            err(f"{where}: paper {paper.get('slug')!r} has role {paper.get('role')!r}")

    # ── checks ─────────────────────────────────────────────────────────────
    checks = spec.get("checks") or []
    if not checks:
        err(f"{where}: no checks — the learner would have no way to verify themselves")
    check_ids = [c.get("id") for c in checks]
    # A duplicate here is worse than it looks: env.check() writes into a dict
    # keyed by id, so the second call overwrites the first and the receipt
    # reports one result where the workshop declared two.
    for duplicate in {i for i in check_ids if check_ids.count(i) > 1}:
        err(f"{where}: duplicate check id '{duplicate}' — one result would overwrite the other")
    for check in checks:
        if check.get("min") is None and check.get("max") is None:
            err(f"{where}: check '{check.get('id')}' declares no threshold")

    # ── assets ─────────────────────────────────────────────────────────────
    for asset in spec.get("assets") or []:
        if not (asset.get("sources") or asset.get("url")):
            err(f"{where}: asset {asset.get('name')!r} declares no source")
        if not asset.get("sha256"):
            warn(f"{where}: asset {asset.get('name')!r} is unpinned (sha256: null)")
        if len(asset.get("sources") or []) < 2:
            warn(f"{where}: asset {asset.get('name')!r} has no mirror — one 404 from broken")

    # ── the thumbnail ──────────────────────────────────────────────────────
    if not any(b.get("thumbnail") for b in body):
        warn(f"{where}: no cell marked thumbnail — the index card will fall back to a tint")

    # ── the notebooks ──────────────────────────────────────────────────────
    for lang in ("en", "ar"):
        nb_path = NOTEBOOKS / f"{slug}.{lang}.ipynb"
        if not nb_path.exists():
            err(f"generated/notebooks/{slug}.{lang}.ipynb is missing — run the build")
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

        for i, cell in enumerate(cells):
            if not cell.get("id"):
                err(f"{nb_path.name}: cell {i} has no id (nbformat >= 4.5 requires one)")
            if cell["cell_type"] != "code":
                continue
            if cell.get("outputs"):
                err(f"{nb_path.name}: cell {i} ships committed outputs — they must be stripped")
            if cell.get("execution_count") is not None:
                err(f"{nb_path.name}: cell {i} has a non-null execution_count")

        notebook_ids = {
            c["metadata"]["azimuth"]["cellId"]
            for c in cells
            if c["cell_type"] == "code" and "azimuth" in c.get("metadata", {})
        }
        for missing in sorted(set(ids) - notebook_ids):
            err(f"{nb_path.name}: declared cell '{missing}' is not in the notebook")

        meta = nb.get("metadata", {}).get("azimuth", {})
        if meta.get("lang") != lang:
            err(f"{nb_path.name}: metadata says lang={meta.get('lang')!r}")
        if meta.get("slug") != slug:
            err(f"{nb_path.name}: metadata says slug={meta.get('slug')!r}")


def main() -> int:
    directories = sorted(d for d in WORKSHOPS.iterdir() if d.is_dir())
    if not directories:
        print("no workshops found")
        return 1
    for directory in directories:
        check_workshop(directory)

    for warning in warnings:
        print(f"  ⚠ {warning}")
    for error in errors:
        print(f"  ✖ {error}")

    print(
        f"\n{len(directories)} workshop(s) · {len(errors)} error(s) · {len(warnings)} warning(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
