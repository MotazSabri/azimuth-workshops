#!/usr/bin/env python3
"""Pin a workshop's assets: download, hash, and write the hash into its YAML.

    python tools/pin-asset.py anomaly-detection-autoencoder

An unpinned asset is the quietest failure mode in a notebook corpus. When a
URL you do not control starts returning something else — a redesigned CSV, a
login page, an HTML 404 — nothing raises. pandas reads the error page as a
one-column frame, the shapes almost work, and the workshop trains on garbage
while printing plausible numbers. A hash turns that silent corruption into
AZ-E302 and a one-line fix.

This writes the hash in place with a line-level edit rather than re-serializing
the YAML, because round-tripping through a YAML library would strip every
comment in the file — and in this repository the comments carry the reasoning.
"""

from __future__ import annotations

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 120


def pin(slug: str) -> int:
    path = ROOT / "workshops" / slug / "workshop.yaml"
    if not path.exists():
        print(f"no such workshop: {slug}")
        return 1

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    changed = 0

    for asset in spec.get("assets") or []:
        name = asset["name"]
        if asset.get("sha256"):
            print(f"  · {name} — already pinned")
            continue

        digest = None
        for url in asset.get("sources") or []:
            try:
                with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                    data = response.read()
            except Exception as exc:
                print(f"  · {name} — {url} failed ({type(exc).__name__})")
                continue
            digest = hashlib.sha256(data).hexdigest()
            print(f"  · {name} — {len(data):,} bytes from {url}")
            print(f"    sha256: {digest}")
            print(f"    bytes:  {len(data)}")
            break

        if digest is None:
            print(f"  ✖ {name} — every source failed; not pinned")
            return 1

        # Replace the `sha256: null` line that belongs to THIS asset: the first
        # one at or after the asset's `name:` line. Anchoring on the name is
        # what keeps a multi-asset workshop from having its hashes swapped.
        lines = text.split("\n")
        start = next(
            (i for i, line in enumerate(lines) if re.match(rf"\s*-?\s*name:\s*{re.escape(name)}\s*$", line)),
            None,
        )
        if start is None:
            print(f"  ✖ {name} — could not find its block in the YAML")
            return 1
        for i in range(start, min(start + 12, len(lines))):
            if re.match(r"\s*sha256:\s*null\s*$", lines[i]):
                indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                lines[i] = f"{indent}sha256: {digest}"
                changed += 1
                break
        else:
            print(f"  ✖ {name} — no `sha256: null` line found near it")
            return 1
        text = "\n".join(lines)

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"\npinned {changed} asset(s) in {path.relative_to(ROOT)}")
    else:
        print("\nnothing to pin")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(pin(sys.argv[1]))
