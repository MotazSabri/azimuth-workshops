#!/usr/bin/env python3
"""Turn an ALREADY-EXECUTED notebook into a run manifest.

    python tools/capture-notebook.py <slug> en path/to/executed_en.ipynb
    python tools/capture-notebook.py <slug> --pair en.ipynb ar.ipynb

WHY THIS EXISTS

`tools/execute.py` both runs a notebook and captures it, which is right when
the machine can run it. Some workshops declare `platforms: [linux]` — the
quantization one needs bitsandbytes, which has no working Windows build — so
on those machines execute.py correctly refuses, and there was then NO
supported route from a Colab run to a manifest. The page would say "outputs
pending re-run" forever while a perfectly good executed notebook sat on disk.

This is the capture half on its own. It reuses execute.py's normalisation, so
a manifest produced here is byte-comparable with one CI produces — a second
capture path that formatted things differently would be worse than none.

It REFUSES a notebook whose cells disagree with the current code.py, because
importing outputs from superseded code is precisely the drift the whole
manifest system exists to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# execute.py owns the output normalisation; import it rather than restate it.
_spec = importlib.util.spec_from_file_location("_ex", ROOT / "tools" / "execute.py")
_ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex)


def capture(slug: str, lang: str, notebook_path: Path, force: bool) -> int:
    spec_path = ROOT / "workshops" / slug / "workshop.yaml"
    code_path = ROOT / "workshops" / slug / "code.py"
    if not spec_path.exists():
        print(f"no such workshop: {slug}")
        return 1

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    code_hash = hashlib.sha256(code_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    # The notebook stamps the hash it was built from. Importing outputs from a
    # notebook older than code.py would put numbers on the page that no
    # committed code produces — the exact failure the drift assertions exist
    # to catch, arriving through a side door.
    stamped = nb.get("metadata", {}).get("azimuth", {}).get("codeHash")
    if stamped and stamped != code_hash and not force:
        print(f"  x {slug} ({lang}) — notebook built from {stamped}, code.py is {code_hash}")
        print("      rebuild and re-run it, or pass --force if you are certain")
        return 1

    stated_lang = nb.get("metadata", {}).get("azimuth", {}).get("lang")
    if stated_lang and stated_lang != lang:
        print(f"  x {slug} — that file is the '{stated_lang}' notebook, not '{lang}'")
        return 1

    manifest = _ex.capture(slug, lang, nb, elapsed=0.0)
    # elapsed is unknown for an imported run: 0 is honest, a guess is not.
    manifest["receipt"]["elapsedSec"] = None
    manifest["receipt"]["source"] = "imported"

    problems = _ex.assert_no_drift(slug, spec, manifest)
    for problem in problems:
        print(f"  x {problem}")

    out = ROOT / "generated" / "runs" / slug / f"{lang}.manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    figures = sum(1 for outs in manifest["cells"].values() for o in outs if o["kind"] == "image")
    print(
        f"  . {slug} ({lang}) — {manifest['status']} · "
        f"{len(manifest['metrics'])} metric(s) · {figures} figure(s) → {out.relative_to(ROOT)}"
    )
    return 1 if problems or manifest["status"] != "verified" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("lang", nargs="?", choices=["en", "ar"])
    parser.add_argument("notebook", nargs="?")
    parser.add_argument("--pair", nargs=2, metavar=("EN_IPYNB", "AR_IPYNB"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.pair:
        worst = 0
        for lang, path in zip(("en", "ar"), args.pair, strict=True):
            worst = max(worst, capture(args.slug, lang, Path(path), args.force))
        return worst
    if not args.lang or not args.notebook:
        print(__doc__)
        return 2
    return capture(args.slug, args.lang, Path(args.notebook), args.force)


if __name__ == "__main__":
    sys.exit(main())
