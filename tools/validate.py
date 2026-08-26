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


def check_yaml_portability(path: Path, where: str) -> None:
    """Refuse YAML that two parsers would read differently.

    This repository is parsed by PyYAML (the shim, in the notebook) and by
    js-yaml (the site, at build time). PyYAML implements YAML 1.1; js-yaml
    defaults to the YAML 1.2 core schema. They disagree on two constructs that
    look completely ordinary:

        bytes: 7_877_383   ->  int 7877383   vs  str "7_877_383"
        gpu: yes           ->  bool True     vs  str "yes"

    Neither raises. The workshop simply behaves differently depending on which
    side is looking at it, and the divergence surfaces as a budget check that
    passes in the notebook and fails on the site.
    """
    for n, line in enumerate(path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"), 1):
        code = line.split("#", 1)[0]
        if re.search(r":\s*-?\d[\d_]*_[\d_]*\d\s*$", code):
            err(f"{where}:{n}: numeric underscore — PyYAML reads an int, js-yaml a string")
        if re.search(r":\s*(yes|no|on|off|Yes|No|On|Off|YES|NO|ON|OFF)\s*$", code):
            err(f"{where}:{n}: bare yes/no/on/off — a bool to PyYAML, a string to js-yaml")


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

    spec_text = spec_path.read_text(encoding="utf-8")
    spec = yaml.safe_load(spec_text)
    code = code_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # ── the two parsers must agree about this file ─────────────────────────
    # This validator runs under PyYAML (YAML 1.1). The site reads the same
    # file with js-yaml (YAML 1.2). Where the two specs differ, a value means
    # different things on the two sides of the project and nothing raises.
    #
    # Underscore digit separators are the case that actually bit us:
    # `bytes: 7_877_632` is the integer 7877632 to PyYAML and the string
    # "7_877_632" to js-yaml.
    for i, line in enumerate(spec_text.split("\n"), 1):
        if re.search(r":\s*-?\d+(?:_\d+)+\s*(?:#.*)?$", line):
            err(
                f"{where}:{i}: numeric underscore separator — PyYAML reads it as a "
                f"number and js-yaml reads it as a string. Write plain digits."
            )
        # YAML 1.1 also reads unquoted yes/no/on/off as booleans; YAML 1.2
        # does not. Quote them or use true/false.
        if re.search(r":\s*(?:yes|no|on|off|Yes|No|On|Off|YES|NO|ON|OFF)\s*(?:#.*)?$", line):
            err(
                f"{where}:{i}: unquoted yes/no/on/off — a boolean to PyYAML, a "
                f"string to js-yaml. Use true/false."
            )

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

    # ── enums ──────────────────────────────────────────────────────────────
    # These reuse the PAPER enums exactly (types/content.ts). A wrong value
    # does not crash anything — the site simply filters the workshop out of
    # the index, search and the newsletter, and the author is left staring at
    # nine empty constellations wondering what broke. Silence is the whole
    # problem, so it is an error here.
    valid_status = ("stable", "draft")
    if spec.get("status") not in valid_status:
        err(
            f"{where}: status is {spec.get('status')!r} — must be one of "
            f"{' | '.join(valid_status)}. 'published' is a common guess and is NOT a "
            f"value; the site filters on 'stable' and would hide this workshop."
        )
    valid_difficulty = ("foundational", "beginner", "intermediate", "advanced")
    if spec.get("difficulty") not in valid_difficulty:
        err(
            f"{where}: difficulty is {spec.get('difficulty')!r} — must be one of "
            f"{' | '.join(valid_difficulty)}"
        )
    valid_tier = ("stable", "moderate", "volatile")
    if (spec.get("maintenance") or {}).get("tier") not in valid_tier:
        err(f"{where}: maintenance.tier must be one of {' | '.join(valid_tier)}")

    # ── scale prose must not restate the scale ─────────────────────────────
    # Same rule as check labels, and for the same reason: the note read
    # "60 epochs" for weeks after the profile moved to 160, telling readers
    # the wrong number directly above code that would run the right one.
    # Use {{scale.epochs}} — the site resolves it from the default profile.
    for i, block in enumerate(body):
        if block.get("type") != "scaleNote":
            continue
        for lang, text in (block.get("text") or {}).items():
            if not isinstance(text, str):
                continue
            stripped = re.sub(r"\{\{[^}]+\}\}", "", text)
            if re.search(r"(?<![A-Za-z])\d+(?:\.\d+)?", stripped):
                warn(
                    f"{where}: body[{i}] scaleNote.{lang} hardcodes a number — "
                    f"use {{{{scale.<knob>}}}} so it cannot drift from the profile"
                )

    # ── chronology ─────────────────────────────────────────────────────────
    if spec.get("status") == "stable" and not spec.get("publishedAt"):
        err(
            f"{where}: status is stable but publishedAt is unset — the newsletter "
            f"window is a date comparison, so this workshop would never appear in one"
        )

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
        # A label that restates the threshold is a second copy of it, and the
        # copy does not move when the bar does. This exact drift shipped once:
        # `min` went 2.0 -> 2.5 and the label still read "at least 2x".
        for lang, text in (check.get("label") or {}).items():
            # A digit bound to a letter is part of a name ("F1", "R2", "GPT-4"),
            # not a threshold. Only free-standing numbers are the drift risk.
            if isinstance(text, str) and re.search(r"(?<![A-Za-z])\d+(?:\.\d+)?", text):
                warn(
                    f"{where}: check '{check.get('id')}' label.{lang} contains a number — "
                    f"env.check() already prints the threshold from min/max"
                )

    # ── assets ─────────────────────────────────────────────────────────────
    for asset in spec.get("assets") or []:
        if not (asset.get("sources") or asset.get("url")):
            err(f"{where}: asset {asset.get('name')!r} declares no source")
        if not asset.get("sha256"):
            warn(f"{where}: asset {asset.get('name')!r} is unpinned (sha256: null)")
        # A single source is only a risk when it is SOMEONE ELSE'S. An asset
        # served from this repository is as available as the workshop itself,
        # so warning about it would be noise the author learns to ignore —
        # and a warning people ignore stops protecting the ones that matter.
        sources = asset.get("sources") or []
        ours = [s for s in sources if "azimuth-workshops" in s]
        if len(sources) < 2 and not ours:
            warn(
                f"{where}: asset {asset.get('name')!r} has a single third-party source — "
                f"one 404 from broken; add a mirror under mirror/"
            )

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

    print(f"\n{len(directories)} workshop(s) · {len(errors)} error(s) · {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
