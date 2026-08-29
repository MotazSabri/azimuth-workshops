#!/usr/bin/env python3
"""Execute a workshop's notebooks and capture their outputs.

    python tools/execute.py                    # every workshop, both languages
    python tools/execute.py <slug>             # one workshop
    python tools/execute.py <slug> --lang en   # one language
    python tools/execute.py <slug> --dry-run   # build the exec copy, do not run

Writes, per workshop and language:

    generated/runs/<slug>/<lang>.manifest.json   the join the site reads
    generated/runs/<slug>/<name>.svg|png         captured figures

WHY nbclient AND NOT papermill

The plan called for papermill. papermill's own execution engine IS nbclient --
what it adds on top is parameter injection and progress reporting. We need
neither: parameters arrive by rewriting the bootstrap cell (which has to be
rewritten anyway, because CI must not `git clone` the repository it is already
standing in), and CI logs are the progress report. So papermill would be one
more pinned dependency wrapping a dependency we already have. If it ever earns
its place -- parameter sweeps across profiles, say -- swapping it in touches
only `run_notebook` below.

WHAT THE EXECUTION COPY CHANGES, AND WHY EACH CHANGE IS SAFE

The notebook CI runs is not byte-identical to the one a learner opens. Three
edits, none of which touch the workshop's own code:

  1. The bootstrap cell is rewritten to point at the local checkout instead of
     cloning. Cloning inside CI would execute a DIFFERENT commit than the one
     being verified -- the whole point of the run is to vouch for this tree.
  2. `InlineBackend.figure_formats = ['svg']` is set. SVG survives dark mode
     and zoom; a rasterized plot on a night-realm page does not.
  3. A capture cell is appended, which serializes the values named in each
     block's `captures:` and the receipt the shim wrote.

None of these are in `code.py`, so `codeHash` -- computed over `code.py` alone
-- is unaffected. That is what lets the manifest's hash vouch for the code the
site displays.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSHOPS = ROOT / "workshops"
NOTEBOOKS = ROOT / "generated" / "notebooks"
RUNS = ROOT / "generated" / "runs"

CAPTURE_TAG = "__azimuth_capture__"
#: env.receipt() prints this, so provenance survives a run executed outside
#: this tool. Must mirror azimuth_nb.RECEIPT_TAG.
RECEIPT_TAG = "__azimuth_receipt__"

# Cells the BUILDER injects, which no workshop.yaml declares and none should.
# They carry cellIds so the executor can find and rewrite them, but they are
# not content — leaving them out of this set made the drift assertion report
# "outputs for 'dependencies', which the YAML no longer declares" on every run
# of every workshop that has dependencies, which is a correct sentence about
# the wrong thing.
INJECTED = ("bootstrap", "dependencies", "__capture__")


# -- the injected cells ------------------------------------------------------


def bootstrap_source(slug: str, lang: str) -> str:
    """Replaces the learner's clone-and-find cell.

    CI is already standing in the tree it is verifying. Cloning here would
    fetch `stable` -- a DIFFERENT commit -- and the run would vouch for code
    nobody is reviewing.
    """
    return (
        "# -- rewritten by tools/execute.py --------------------------------\n"
        "# CI runs against the checkout it is verifying, never a fresh clone:\n"
        "# cloning would execute a different commit than the one under review.\n"
        "%matplotlib inline\n"
        "%config InlineBackend.figure_formats = ['svg']\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"SLUG = {slug!r}\n"
        f"LANG = {lang!r}\n"
        'PROFILE = "free"\n'
        f"root = Path({str(ROOT)!r})\n"
        "sys.path.insert(0, str(root / 'shim'))\n"
        "os.environ.setdefault('AZIMUTH_DATA_DIR', str(root / 'data'))"
    )


def capture_source(captures: dict) -> str:
    """Appended last. Serializes declared captures and the shim's receipt.

    Only SCALARS become metrics. `loss_curve` is 160 floats -- a real capture,
    but not a number you can interpolate into a sentence, and putting it in
    `metrics` would invite `{{run.metrics.loss_curve}}` to render as a wall of
    digits. Non-scalars are recorded by name and shape, so the manifest shows
    they were produced without pretending they are metrics.
    """
    return f'''# -- appended by tools/execute.py ---------------------------------
# Not part of the workshop. Serializes what the YAML declared as `captures:`.
import json as _json

_declared = {captures!r}
_metrics, _shapes = {{}}, {{}}
for _cell, _names in _declared.items():
    for _n in _names:
        if _n not in globals():
            continue
        _v = globals()[_n]
        if isinstance(_v, (bool, int, float, str)):
            _metrics[_n] = _v
        else:
            try:
                _shapes[_n] = f"{{type(_v).__name__}}[{{len(_v)}}]"
            except TypeError:
                _shapes[_n] = type(_v).__name__

_receipt = {{}}
try:
    with open("azimuth-receipt.json", encoding="utf-8") as _fh:
        _receipt = _json.load(_fh)
except OSError:
    pass

print("{CAPTURE_TAG}" + _json.dumps(
    {{"metrics": _metrics, "shapes": _shapes, "receipt": _receipt}}, default=str))
'''


# -- build the execution copy ------------------------------------------------


def build_exec_notebook(slug: str, lang: str, spec: dict) -> dict:
    path = NOTEBOOKS / f"{slug}.{lang}.ipynb"
    if not path.exists():
        raise SystemExit(f"{path.name} is missing -- run tools/build_notebooks.py first")
    nb = json.loads(path.read_text(encoding="utf-8"))

    captures = {
        b["id"]: b["captures"] for b in spec.get("body", []) if b.get("captures") and b.get("id")
    }

    for cell in nb["cells"]:
        if cell.get("metadata", {}).get("azimuth", {}).get("cellId") == "bootstrap":
            lines = bootstrap_source(slug, lang).split("\n")
            cell["source"] = [f"{ln}\n" for ln in lines[:-1]] + [lines[-1]]

    nb["cells"].append(
        {
            "cell_type": "code",
            "id": "az-99-capture",
            "metadata": {"azimuth": {"cellId": "__capture__"}},
            "execution_count": None,
            "outputs": [],
            "source": capture_source(captures),
        }
    )
    return nb


# -- run ---------------------------------------------------------------------


def run_notebook(nb: dict, timeout: int):
    """Execute in place. Returns (notebook, fatal error or None).

    `allow_errors=True` on purpose: a failed run must still produce a manifest
    with its error captured, because a page that silently omits a cell cannot
    tell the reader whether it produced nothing or produced a failure.
    """
    import nbformat
    from nbclient import NotebookClient

    # reads(), not from_dict(): nbformat stores `source` as a list of lines,
    # and only the reader rejoins them into the string nbclient expects.
    # from_dict passes the list straight through and every cell dies on
    # `cell.source.strip()`.
    doc = nbformat.reads(json.dumps(nb), as_version=4)
    client = NotebookClient(
        doc,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    except Exception as exc:
        return json.loads(nbformat.writes(doc)), f"{type(exc).__name__}: {exc}"
    return json.loads(nbformat.writes(doc)), None


# -- capture -----------------------------------------------------------------

SVG_MIME = "image/svg+xml"
PNG_MIME = "image/png"

# The receipt cell PRINTS the completion code, and captured stream output is
# published twice over: committed to this public repo, and rendered on the
# workshop page. Publishing it would hand every reader the answer to the one
# check that says they ran the thing.
#
# The site stores only a digest of this code for exactly that reason, so
# letting the plaintext ride along in a stream output would defeat the design
# from the other end. Found by rendering the page and searching for it.
COMPLETION_CODE = re.compile(r"AZ-[0-9A-F]{10}")

# Progress bars, download meters and library warnings are runtime furniture,
# not results. They are legitimately useful in a live notebook and pure noise
# on a published page — the tokenizers cell emitted four tqdm widgets and the
# same HF auth warning twice, once as a stream and once through logging.
#
# Dropped at capture rather than silenced in code.py: suppressing them in the
# workshop would ALSO hide them from the learner, who has a real reason to see
# a download is running. The page is the only place they are worthless.
NOISE = re.compile(
    r"""^\s*(
        .*IProgress\ not\ found
      | .*huggingface_hub.*
      | Warning:\ You\ are\ sending\ unauthenticated\ requests
      | .*TqdmWarning.*
      | \s*from\ \.autonotebook\ import\ tqdm.*
      | .*\d+%\|[\u2588\u2589\u258a\u258b\u258c\u258d\u258e\u258f\s|]*\|.*
    )""",
    re.VERBOSE,
)


def strip_noise(text: str) -> str:
    """Drop progress/warning lines, keep everything the workshop printed."""
    kept = [ln for ln in text.split("\n") if not NOISE.match(ln)]
    return "\n".join(kept).strip("\n")


REDACTED = "AZ-\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588"


def normalize_outputs(cell_id: str, outputs: list, out_dir: Path, slug: str) -> list:
    """Map nbformat outputs onto the five kinds the site renders.

    Everything nbformat emits collapses into: stream, image, table, metric,
    error. A sixth kind would mean a sixth renderer on the site, so anything
    that does not fit is dropped rather than passed through -- an unrenderable
    output is worse than an absent one.
    """
    captured: list = []
    figure_n = 0

    for output in outputs:
        kind = output.get("output_type")

        if kind == "stream":
            text = "".join(output.get("text", []))
            if CAPTURE_TAG in text:  # the injected cell's payload, not content
                continue
            # Drop only the tagged LINE, not the whole stream: the receipt is
            # printed alongside "Workshop complete." and the completion code,
            # which the reader is meant to see.
            if RECEIPT_TAG in text:
                text = "\n".join(ln for ln in text.split("\n") if RECEIPT_TAG not in ln)
            text = strip_noise(text)
            if text.strip():
                captured.append(
                    {"kind": "stream", "text": COMPLETION_CODE.sub(REDACTED, text.rstrip())}
                )

        elif kind in ("display_data", "execute_result"):
            data = output.get("data", {})
            if SVG_MIME in data or PNG_MIME in data:
                figure_n += 1
                svg = SVG_MIME in data
                suffix = "" if figure_n == 1 else f"-{figure_n}"
                name = f"{cell_id}{suffix}.{'svg' if svg else 'png'}"
                blob = data[SVG_MIME] if svg else data[PNG_MIME]
                payload = "".join(blob) if isinstance(blob, list) else blob
                target = out_dir / name
                if svg:
                    target.write_text(payload, encoding="utf-8")
                else:
                    target.write_bytes(base64.b64decode(payload))
                captured.append(
                    {
                        # `src` is relative to the RUNS ROOT -- `<slug>/<file>`,
                        # never prefixed with `runs/`. The site resolves it as
                        # /generated/runs/<src>.
                        "kind": "image",
                        "src": f"{slug}/{name}",
                        "format": "svg" if svg else "png",
                    }
                )
            elif "widget-view" in " ".join(data):
                # A tqdm/ipywidgets view. Its text/plain fallback is a progress
                # bar snapshot frozen at whatever percentage the run happened to
                # end on — meaningless on a page.
                continue
            elif "text/plain" in data:
                text = strip_noise("".join(data["text/plain"])).strip()
                if text:
                    captured.append({"kind": "stream", "text": COMPLETION_CODE.sub(REDACTED, text)})

        elif kind == "error":
            captured.append(
                {
                    "kind": "error",
                    "ename": output.get("ename", "Error"),
                    "evalue": output.get("evalue", ""),
                }
            )

    # Coalesce consecutive streams. A loop that prints once per epoch emits one
    # nbformat output per flush -- the training cell produced seventeen -- and
    # each would render as its own bordered box. They are one log; the page
    # should show one block.
    merged: list = []
    for output in captured:
        if output["kind"] == "stream" and merged and merged[-1]["kind"] == "stream":
            merged[-1]["text"] += "\n" + output["text"]
        else:
            merged.append(output)
    return merged


def extract_receipt(nb: dict) -> dict:
    """The receipt env.receipt() printed, from anywhere in the notebook.

    This is the only route by which a Colab run carries its device, torch
    version and timing into a manifest — the injected capture cell does not
    exist in a notebook someone ran in a browser.
    """
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            for line in "".join(output.get("text", [])).split("\n"):
                if RECEIPT_TAG in line:
                    try:
                        return json.loads(line.split(RECEIPT_TAG, 1)[1].strip())
                    except json.JSONDecodeError:
                        return {}
    return {}


def extract_payload(nb: dict) -> dict:
    for cell in nb["cells"]:
        if cell.get("metadata", {}).get("azimuth", {}).get("cellId") != "__capture__":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = "".join(output.get("text", []))
            if CAPTURE_TAG in text:
                return json.loads(text.split(CAPTURE_TAG, 1)[1].strip())
    return {}


def git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()
    except OSError:
        return ""


def capture(slug: str, lang: str, nb: dict, elapsed: float) -> dict:
    out_dir = RUNS / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = extract_payload(nb)
    # The injected cell wins when present (it also carries metrics); the
    # printed line is the fallback for an imported run.
    if not payload.get("receipt"):
        payload = {**payload, "receipt": extract_receipt(nb)}
    cells: dict = {}
    failed = False

    for cell in nb["cells"]:
        cell_id = cell.get("metadata", {}).get("azimuth", {}).get("cellId")
        if not cell_id or cell_id in INJECTED:
            continue
        outputs = normalize_outputs(cell_id, cell.get("outputs", []), out_dir, slug)
        if any(o["kind"] == "error" for o in outputs):
            failed = True
        cells[cell_id] = outputs

    # Hashed from NORMALIZED bytes, so a Windows checkout and a Linux CI
    # runner agree. Hashing raw bytes would make every codeHash differ by
    # platform, and the drift assertion would fire on every page forever.
    code_bytes = (WORKSHOPS / slug / "code.py").read_bytes().replace(b"\r\n", b"\n")
    code_hash = hashlib.sha256(code_bytes).hexdigest()[:16]
    receipt = payload.get("receipt", {})

    return {
        "schema": "run/v1",
        "slug": slug,
        "lang": lang,
        "codeHash": code_hash,
        # `verified` only when nothing errored. Anything else and the site
        # shows the code without vouching for the numbers.
        "status": "failed" if failed else "verified",
        "profile": receipt.get("profile", "free"),
        "receipt": {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": receipt.get("device", "unknown"),
            "torch": receipt.get("torch", "unknown"),
            "python": receipt.get("python", sys.version.split()[0]),
            "shim": receipt.get("shim", "unknown"),
            "elapsedSec": round(elapsed, 1),
            "commit": git("rev-parse", "HEAD") or "unknown",
        },
        "metrics": payload.get("metrics", {}),
        "shapes": payload.get("shapes", {}),
        "cells": cells,
    }


# -- drift, checked here too -------------------------------------------------


def assert_no_drift(slug: str, spec: dict, manifest: dict) -> list:
    """The same assertions the site makes.

    Duplicated deliberately, and this is the one duplication in this project
    that earns its keep: the site's copy protects the READER from stale
    numbers; this one stops a stale manifest being committed at all. Catching
    it here makes the failure a red CI run rather than a page that quietly
    degrades to "pending" and waits for someone to notice.
    """
    problems = []
    declared = {b["id"] for b in spec.get("body", []) if b.get("type") in ("cell", "exercise")}
    captured = set(manifest["cells"]) - set(INJECTED)
    for missing in sorted(declared - captured):
        problems.append(f"{slug}: cell '{missing}' produced no outputs")
    for orphan in sorted(captured - declared):
        if orphan == "__run__":
            continue
        problems.append(f"{slug}: outputs for '{orphan}', which the YAML no longer declares")
    return problems


# -- main --------------------------------------------------------------------


def unsupported_here(spec: dict) -> str | None:
    """The platform this machine is not, or None if it can run this workshop.

    Checked BEFORE a kernel starts. The shim raises AZ-E104 for the same
    reason, but by then a dependency install has already run — thirty seconds
    to be told the answer was knowable from a YAML field.
    """
    profile = next(
        (p for p in spec.get("profiles") or [] if p.get("default")),
        (spec.get("profiles") or [None])[0],
    )
    platforms = ((profile or {}).get("requires") or {}).get("platforms")
    if not platforms:
        return None
    current = (
        "linux"
        if sys.platform.startswith("linux")
        else "macos"
        if sys.platform == "darwin"
        else "windows"
        if sys.platform.startswith("win")
        else sys.platform
    )
    wanted = {str(x).strip().lower() for x in platforms}
    return None if current in wanted else f"{'/'.join(sorted(wanted))}, this is {current}"


def execute_one(slug: str, lang: str, timeout: int, dry_run: bool):
    spec = yaml.safe_load((WORKSHOPS / slug / "workshop.yaml").read_text(encoding="utf-8"))

    # SKIP, DO NOT FAIL, and above all do not write a manifest.
    #
    # A machine that cannot run a workshop has learned nothing about it. The
    # first version wrote a `failed` manifest anyway — which would have
    # overwritten a VERIFIED capture from Colab or a Linux runner the next
    # time someone ran this locally, destroying a good result because the
    # wrong machine tried. Silence is the correct output here.
    reason = unsupported_here(spec)
    if reason:
        print(f"  - {slug} ({lang}) -- skipped: needs {reason}")
        return None, []
    nb = build_exec_notebook(slug, lang, spec)

    if dry_run:
        target = RUNS / slug / f"{lang}.exec.ipynb"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  . {slug} ({lang}) -- dry run, wrote {target.relative_to(ROOT)}")
        return True, []

    started = time.time()
    executed, fatal = run_notebook(nb, timeout)
    elapsed = time.time() - started

    manifest = capture(slug, lang, executed, elapsed)
    if fatal:
        manifest["status"] = "failed"
        manifest["cells"]["__run__"] = [
            {"kind": "error", "ename": "ExecutionError", "evalue": fatal}
        ]

    problems = assert_no_drift(slug, spec, manifest)

    path = RUNS / slug / f"{lang}.manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    figures = sum(1 for outs in manifest["cells"].values() for o in outs if o["kind"] == "image")
    print(
        f"  . {slug} ({lang}) -- {manifest['status']} in {elapsed:.0f}s . "
        f"{len(manifest['metrics'])} metric(s) . {figures} figure(s)"
    )

    # A bare "failed in 24s" sends you to open a JSON file to find out what
    # broke. The FIRST error is almost always the cause and everything after
    # it is fallout, so print that one and say where the rest are.
    if manifest["status"] == "failed":
        for cell_id, outputs in manifest["cells"].items():
            first = next((o for o in outputs if o["kind"] == "error"), None)
            if first:
                detail = first["evalue"].strip().split("\n")[0][:160]
                print(f"      first failure in [{cell_id}]: {first['ename']}: {detail}")
                break
        print(f"      full log: generated/runs/{slug}/{lang}.manifest.json")
    return manifest["status"] == "verified" and not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?")
    parser.add_argument("--lang", choices=["en", "ar"])
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    langs = [args.lang] if args.lang else config["build"]["languages"]
    slugs = [args.slug] if args.slug else sorted(d.name for d in WORKSHOPS.iterdir() if d.is_dir())

    for slug in slugs:
        if not (WORKSHOPS / slug).is_dir():
            print(f"no such workshop: {slug}")
            return 1

    ok = True
    skipped = 0
    all_problems: list = []
    for slug in slugs:
        for lang in langs:
            passed, problems = execute_one(slug, lang, args.timeout, args.dry_run)
            if passed is None:  # skipped: not runnable on this platform
                skipped += 1
                continue
            ok = ok and passed
            all_problems += problems

    for problem in all_problems:
        print(f"  x {problem}")

    if skipped:
        print(
            f"\n{skipped} run(s) skipped -- this platform cannot run them, and no "
            "manifest was written.\nOpen the workshop in Colab, or let the `execute` "
            "workflow run it on a Linux runner."
        )
    if not ok and not args.dry_run:
        print("\nrun did not verify -- manifests written, but stable must not advance")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
