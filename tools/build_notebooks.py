#!/usr/bin/env python3
"""Build every workshop's notebooks from workshop.yaml + code.py.

    python tools/build_notebooks.py              # write generated/notebooks/
    python tools/build_notebooks.py --check      # fail if anything would change
    python tools/build_notebooks.py <slug>       # one workshop

WHY THIS IS PYTHON AND NOT .mjs

The plan called for `build-workshops.mjs`. That fit when the builder was
assumed to live in the site repository, where it could share the sanitizers in
scripts/lib/content-utils.mjs. The dependency runs the other way: the site
consumes THIS repository through a submodule, so nothing here can import from
there. The builder's whole job is local — YAML plus a .py into .ipynb — and
this repository is otherwise single-toolchain (PyYAML for the shim, ruff for
linting, no package.json). A .mjs builder would mean CI installs Node as well
as Python, and a second dependency manifest to keep pinned, to share nothing.
The sanitizers stay shared where they are actually needed: the site rendering
workshop prose.

DETERMINISM IS THE POINT

A one-word prose fix must produce a one-line diff. That rules out anything
incidental leaking into the output — timestamps, random cell ids (nbformat's
default), hash-derived ids that churn on every edit, dict iteration order. Cell
ids are position plus the authored cell id, and nothing else varies. `--check`
rebuilds in memory and compares against what is committed, so CI can prove the
notebooks match their source without trusting anyone to have run the build.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSHOPS = ROOT / "workshops"
NOTEBOOKS = ROOT / "generated" / "notebooks"

#: Highest code point that is unambiguously ASCII. Identifiers above it are
#: legal Python and refused anyway — see check_identifiers_ascii.
ASCII_MAX = 127

REGION_RE = re.compile(r"^# --8<-- \[start:(\w+)\]\n(.*?)^# --8<-- \[end:\1\]", re.M | re.S)

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


# ── config ───────────────────────────────────────────────────────────────────


def load_config() -> dict[str, Any]:
    path = ROOT / "config.yaml"
    if not path.exists():
        raise SystemExit("config.yaml is missing — the builder has no repo to point notebooks at")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ── source loading ───────────────────────────────────────────────────────────


def extract_regions(code: str, where: str) -> dict[str, str]:
    """Named regions from code.py.

    Anchored to line starts so a comment DESCRIBING the marker syntax cannot
    register as a region — that ambiguity produced a phantom region called
    'name' the first time this was checked by hand.
    """
    regions: dict[str, str] = {}
    for name, body in REGION_RE.findall(code):
        if name in regions:
            err(f"{where}: region '{name}' is defined twice")
        regions[name] = body.strip("\n")
    return regions


def check_identifiers_ascii(source: str, ref: str, where: str) -> None:
    """Refuse non-ASCII IDENTIFIERS. Comments and strings may be Arabic.

    Python 3 permits Unicode identifiers, so `متوسط = 5` runs. It should still
    be refused: the two language builds share one code.py, and an identifier
    that reads naturally in one build is opaque in the other. Worse, Arabic
    identifiers make a traceback unsearchable for the reader most likely to
    need to search for it.

    Parsed with ast rather than regexed, so a comment or a string containing
    Arabic — which is most of this corpus — cannot be mistaken for code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return  # the ruff pass reports syntax errors with better messages

    def bad(name: str) -> bool:
        return any(ord(ch) > ASCII_MAX for ch in name)

    # Deduplicated: one variable appears as both a store and a load target, so
    # walking naively reports the same name once per mention. A reader fixing
    # one identifier should see one error.
    seen: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.keyword) and node.arg:
            names = [node.arg]
        for name in names:
            if bad(name) and name not in seen:
                seen.add(name)
                shown = unicodedata.normalize("NFC", name)
                err(f"{where}: region '{ref}' has a non-ASCII identifier {shown!r}")


def run_ruff(paths: list[Path]) -> None:
    """Lint and format-check the spines. Ruff's formatter is Black-compatible,
    so one tool covers what the plan asked two for — and one fewer pinned
    dependency is one fewer thing to rot."""
    for args, label in ((["check"], "ruff check"), (["format", "--check"], "ruff format")):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *args, *[str(p) for p in paths]],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,  # a lint failure is reported, not raised
        )
        if result.returncode != 0:
            detail = (result.stdout + result.stderr).strip().splitlines()
            for line in detail[:12]:
                err(f"{label}: {line}")


# ── notebook assembly ────────────────────────────────────────────────────────


class Builder:
    """Assembles one workshop, in one language."""

    def __init__(self, slug: str, spec: dict, regions: dict[str, str], config: dict, lang: str):
        self.slug = slug
        self.spec = spec
        self.regions = regions
        self.config = config
        self.lang = lang
        self.ar = lang == "ar"
        self.cells: list[dict] = []
        self.n = 0

    # Bilingual pick. Falls back to English rather than emitting an empty
    # string: a missing `ar` is a lint error elsewhere, and a blank cell in the
    # notebook would hide it.
    def t(self, node: dict | None) -> str:
        if not node:
            return ""
        return node.get(self.lang) or node.get("en", "")

    def _id(self, cid: str | None) -> str:
        """Position plus the authored id. Deterministic by construction:
        nbformat's default is random, and a content hash would churn on every
        prose edit — both break the one-word-fix, one-line-diff rule."""
        self.n += 1
        return f"az-{self.n:02d}-{cid}" if cid else f"az-{self.n:02d}"

    @staticmethod
    def _lines(text: str) -> list[str]:
        """nbformat stores source as a list of lines, each keeping its newline
        except the last."""
        parts = text.rstrip("\n").split("\n")
        return [p + "\n" for p in parts[:-1]] + [parts[-1]]

    def md(self, text: str, cid: str | None = None) -> None:
        cell: dict[str, Any] = {
            "cell_type": "markdown",
            "id": self._id(cid),
            "metadata": {},
            "source": self._lines(text),
        }
        if cid:
            cell["metadata"]["azimuth"] = {"cellId": cid}
        self.cells.append(cell)

    def code(self, text: str, cid: str | None = None) -> None:
        cell: dict[str, Any] = {
            "cell_type": "code",
            "id": self._id(cid),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": self._lines(text),
        }
        if cid:
            cell["metadata"]["azimuth"] = {"cellId": cid}
        self.cells.append(cell)

    # ── the parts ───────────────────────────────────────────────────────────

    def paper_url(self, slug: str) -> str:
        site = self.config["site"]
        return site["base"] + site["paperPath"].format(lang=self.lang, slug=slug)

    def header(self) -> None:
        spec, ar = self.spec, self.ar
        core = [p for p in spec["papers"] if p["role"] == "core"]
        papers = "\n".join(
            f"- [{p['slug']}]({self.paper_url(p['slug'])}) — {self.t(p['label'])}" for p in core
        )
        est = spec["estimate"]
        gpu = spec["profiles"]
        default = next((p for p in gpu if p.get("default")), gpu[0])
        needs_gpu = default["requires"]["gpu"]
        if ar:
            chip = "معالج رسوميات" if needs_gpu else "معالج رسوميات اختياري"
            meta = f"{chip} · ~{est['totalMinutes']} دقيقة · Colab"
            goal_h, papers_h = "الهدف", "الأوراق وراء هذه الورشة"
            save = (
                "> احفظ نسخة في Drive قبل أن تبدأ (ملف ← حفظ نسخة في Drive). "
                "التعديلات على الأصل لا تُحفظ."
            )
        else:
            chip = "GPU" if needs_gpu else "GPU optional"
            meta = f"{chip} · ~{est['totalMinutes']} min · Colab"
            goal_h, papers_h = "The goal", "The papers behind this"
            save = (
                "> Save a copy to Drive before you start (File → Save a copy in Drive). "
                "Edits to the original are not saved."
            )
        self.md(
            f"# {self.t(spec['title'])}\n\n"
            f"**{self.t(spec['concept'])}** · {meta}\n\n"
            f"{self.t(spec['scenario'])}\n\n"
            f"### {goal_h}\n\n{self.t(spec['goal'])}\n\n"
            f"### {papers_h}\n\n{papers}\n\n"
            f"{save}"
        )

    def bootstrap(self, code_hash: str) -> None:
        repo = self.config["repo"]
        if self.ar:
            heading = (
                "## الإعداد\n\n`PROFILE` هو المقبض الوحيد للحجم. المستوى المجاني هو "
                "الافتراضي ويعمل داخل حدود Colab المجانية."
            )
        else:
            heading = (
                "## Setup\n\n`PROFILE` is the only scale knob. The free tier is the "
                "default and stays inside Colab's free envelope."
            )
        self.md(heading)

        dirname = repo["dirname"]
        self.code(
            f'SLUG = "{self.slug}"\n'
            f'LANG = "{self.lang}"\n'
            f'PROFILE = "free"  # free | a100\n'
            "\n"
            "# Colab defaults to inline figures; CI does not. Being explicit means the\n"
            "# captured plot on the site and the plot you see are produced the same way.\n"
            "%matplotlib inline\n"
            "\n"
            "# The shim lives in this repository, not on PyPI: a workshop should never\n"
            "# depend on a package index staying up.\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "# Guarded three ways. Colab users re-run the setup cell constantly, and\n"
            "# a contributor may already be sitting inside a checkout — the first\n"
            "# Windows run of this notebook cloned the repository into its own\n"
            "# generated/notebooks/ directory because neither case was handled.\n"
            "# The hash of the code.py THESE CELLS were built from. setup()\n"
            "# compares it with the code.py it finds on disk: if a notebook is\n"
            "# older than its source, every number below describes code that is\n"
            "# not the code anyone is reading. The printed `code ·` line was\n"
            "# taken from disk and so could not catch this by itself.\n"
            f"os.environ['AZIMUTH_NOTEBOOK_CODEHASH'] = {code_hash!r}\n\n"
            f"REPO = {dirname!r}\n"
            "here = Path.cwd().resolve()\n"
            "root = next((p for p in [here, *here.parents] "
            "if (p / 'shim' / 'azimuth_nb').is_dir()), None)\n"
            "if root is None:\n"
            "    if not Path(REPO).exists():\n"
            "        subprocess.run(\n"
            f"            ['git', 'clone', '--depth', '1', '--branch', {repo['branch']!r},\n"
            f"             {repo['url']!r}, REPO],\n"
            "            check=True,\n"
            "        )\n"
            "    root = (here / REPO).resolve()\n"
            "print('workshop root:', root)\n"
            "\n"
            "# Absolute, so nothing depends on the working directory. Dropping the\n"
            "# %cd this cell used to do also means the notebook stops caring where\n"
            "# it was opened from.\n"
            "sys.path.insert(0, str(root / 'shim'))\n"
            # Assigned, not bare: setdefault RETURNS the value, and a bare call as
            # the last line of a cell makes Jupyter print it — the setup cell was
            # ending with a stray "'C:\\\\Users\\\\...\\\\data'" execute_result.
            "_ = os.environ.setdefault('AZIMUTH_DATA_DIR', str(root / 'data'))",
            cid="bootstrap",
        )

    def dependencies(self) -> None:
        """A pip cell, emitted only when the workshop declares packages.

        torch is still ASSERTED, never installed — a second torch over Colab's
        is the reliable way to break a runtime. But a `moderate` or `volatile`
        workshop legitimately needs a library Colab does not ship, and until
        now there was nowhere to say so: the first gymnasium run had to be
        hand-patched in the notebook, which is a change that vanishes on the
        next build.

        Quiet by default (`-q`) and pinned by the author, so a run is
        reproducible rather than "whatever PyPI had that morning".
        """
        deps = self.spec.get("dependencies") or {}
        pip = deps.get("pip") or []
        apt = deps.get("apt") or []
        if not pip and not apt:
            return

        lines = [
            "# Declared by this workshop (dependencies: in workshop.yaml).",
            "# torch is NOT installed here — it is asserted, because a second",
            "# torch over Colab's own will not match the driver.",
        ]
        if apt:
            lines.append(f"!apt-get -qq install -y {' '.join(apt)} > /dev/null")
        if pip:
            lines.append(f"%pip install -q {' '.join(pip)}")
        self.code("\n".join(lines), cid="dependencies")

    def body(self) -> None:
        for block in self.spec["body"]:
            kind = block["type"]
            if kind == "prose":
                self.md(self.t(block["text"]))
            elif kind == "scaleNote":
                label = "الحجم" if self.ar else "On scale"
                self.md(f"> **{label}** — {self.t(block['text'])}")
            elif kind == "paperRef":
                paper = next(p for p in self.spec["papers"] if p["slug"] == block["slug"])
                label = "الورقة" if self.ar else "The paper"
                self.md(
                    f"> **{label}** · [{block['slug']}]({self.paper_url(block['slug'])})"
                    f" — {self.t(paper['label'])}\n>\n> {self.t(block['note'])}"
                )
            elif kind == "exercise":
                label = "تمرين" if self.ar else "Exercise"
                hint = ""
                if block.get("hint"):
                    word = "تلميح متاح" if self.ar else "A hint is available"
                    hint = f"\n\n_{word}: `env.hint({block['hint']})`_"
                self.md(f"### {label} — {block['id']}\n\n{self.t(block['prompt'])}{hint}")
                self.code(self.regions[block["ref"]], cid=block["id"])
            elif kind == "cell":
                if block.get("caption"):
                    self.md(f"_{self.t(block['caption'])}_")
                self.code(self.regions[block["ref"]], cid=block["id"])

    def build(self, code_hash: str) -> str:
        self.header()
        self.bootstrap(code_hash)
        self.dependencies()
        self.body()
        notebook = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
                "colab": {"provenance": [], "toc_visible": True},
                "azimuth": {
                    "slug": self.slug,
                    "lang": self.lang,
                    "schema": self.spec["schema"],
                    "constellation": self.spec["constellation"],
                    "codeHash": code_hash,
                    "shim": self.config["build"]["shimVersion"],
                    "builder": self.config["build"]["builderVersion"],
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


# ── per-workshop driver ──────────────────────────────────────────────────────


def build_workshop(directory: Path, config: dict, check: bool) -> list[tuple[Path, str]]:
    """Build one workshop's notebooks, or return [] if THIS workshop is broken.

    Errors are scoped per workshop. An earlier version tested the global
    `errors` list, so a fault in the alphabetically-first workshop silently
    suppressed the output of every workshop after it — the build reported the
    first one's problem and said nothing at all about the others, which looked
    exactly like "it just didn't build that one".
    """
    before = len(errors)
    slug = directory.name
    where = f"workshops/{slug}"
    spec_path = directory / "workshop.yaml"
    code_path = directory / "code.py"

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    # Normalize line endings before ANY pattern touches this file. git hands
    # it over with CRLF on Windows, and the region markers are anchored on
    # "\n" — on a Windows checkout the extractor found zero regions and every
    # notebook cell came out empty, with nothing but a log line to show for it.
    code = code_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    # Normalized bytes: the same file checked out on Windows (CRLF) and Linux
    # (LF) must hash identically, or every platform disagrees about which code
    # a run vouched for.
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

    if spec.get("id") != slug:
        err(f"{where}: id {spec.get('id')!r} does not match the directory name")

    regions = extract_regions(code, where)
    refs = [b["ref"] for b in spec.get("body", []) if b.get("ref")]

    # Both directions. A dangling ref would emit an empty cell; an orphaned
    # region is code nothing runs, usually left behind by a prose edit.
    for ref in refs:
        if ref not in regions:
            err(f"{where}: cell ref '{ref}' has no matching region in code.py")
    for name in regions:
        if name not in refs:
            err(f"{where}: region '{name}' is never referenced by a cell")

    ids = [b["id"] for b in spec.get("body", []) if b.get("type") in ("cell", "exercise")]
    for duplicate in {i for i in ids if ids.count(i) > 1}:
        err(f"{where}: duplicate cell id '{duplicate}' — captured outputs would collide")
    for block in spec.get("body", []):
        if block.get("captures") and not block.get("id"):
            err(f"{where}: a block with captures has no id to join them on")

    for ref in refs:
        if ref in regions:
            check_identifiers_ascii(regions[ref], ref, where)

    # Only THIS workshop's errors block THIS workshop.
    if len(errors) > before:
        print(f"  ! {slug} — skipped, see errors below")
        return []

    outputs: list[tuple[Path, str]] = []
    for lang in config["build"]["languages"]:
        text = Builder(slug, spec, regions, config, lang).build(code_hash)
        outputs.append((NOTEBOOKS / f"{slug}.{lang}.ipynb", text))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?", help="build one workshop instead of all")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail if the committed notebooks differ from a fresh build",
    )
    args = parser.parse_args()

    config = load_config()

    from azimuth_nb import __version__ as shim_version  # noqa: PLC0415

    if shim_version != config["build"]["shimVersion"]:
        err(
            f"config.yaml says shimVersion {config['build']['shimVersion']} "
            f"but azimuth_nb.__version__ is {shim_version}"
        )

    directories = sorted(d for d in WORKSHOPS.iterdir() if d.is_dir())
    if args.slug:
        directories = [d for d in directories if d.name == args.slug]
        if not directories:
            print(f"no such workshop: {args.slug}")
            return 1

    run_ruff([d / "code.py" for d in directories])

    written = 0
    unchanged = 0
    skipped = 0
    stale: list[str] = []
    for directory in directories:
        outputs = build_workshop(directory, config, args.check)
        if not outputs:
            skipped += 1
        for path, text in outputs:
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current == text:
                # Already byte-identical. Counted, not silent: "0 notebooks
                # written" with nothing else on the line reads like a failure
                # when it is the opposite — every notebook already matches its
                # source, which is exactly what a deterministic build should
                # say on a second run.
                unchanged += 1
                continue
            if args.check:
                stale.append(path.name)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                written += 1

    for message in errors:
        print(f"  ✖ {message}")
    if errors:
        return 1

    if args.check:
        if stale:
            print("  ✖ committed notebooks differ from a fresh build:")
            for name in stale:
                print(f"      {name}")
            print("\n  run: python tools/build_notebooks.py")
            return 1
        print(f"{len(directories)} workshop(s) · notebooks match their source")
        return 0

    # Say how many were skipped as well as how many were written. "2 workshops
    # · 2 notebooks written" reads like success when four were expected.
    parts = [f"{len(directories)} workshop(s)"]
    if written:
        parts.append(f"{written} notebook(s) written")
    if unchanged:
        parts.append(f"{unchanged} already up to date")
    if not written and not unchanged:
        parts.append("nothing built")
    if skipped:
        parts.append(f"{skipped} workshop(s) SKIPPED")
    print(" · ".join(parts))
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "shim"))
    sys.exit(main())
