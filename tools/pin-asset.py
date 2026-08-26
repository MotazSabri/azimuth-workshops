#!/usr/bin/env python3
"""Pin a workshop's assets: download, hash, and write the hash into its YAML.

    python tools/pin-asset.py anomaly-detection-autoencoder
    python tools/pin-asset.py <slug> --force   # re-pin after the file changed

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


def pin(slug: str, force: bool = False) -> int:
    path = ROOT / "workshops" / slug / "workshop.yaml"
    if not path.exists():
        print(f"no such workshop: {slug}")
        return 1

    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    changed = 0

    for asset in spec.get("assets") or []:
        name = asset["name"]
        if asset.get("sha256") and not force:
            # An asset's bytes CAN legitimately change — a corrected encoding,
            # a trimmed corpus, a fixed column. Refusing to re-pin then means
            # the workshop keeps fetching the old file and reporting it as
            # verified, which is exactly how a bad conversion survived two
            # full round-trips.
            print(f"  · {name} — already pinned (use --force to re-pin)")
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
            size = len(data)
            print(f"  · {name} — {len(data):,} bytes from {url}")
            print(f"    sha256: {digest}")
            print(f"    bytes:  {len(data)}")
            break

        if digest is None:
            print(f"  ✖ {name} — every source failed; not pinned")
            return 1

        # Rewrite the sha256 (and bytes) lines belonging to THIS asset.
        #
        # Anchored on the asset's `name:` and bounded by the next list item,
        # rather than by a fixed line count: an asset block with a few lines
        # of comment in it is normal, and a 12-line window silently missed
        # one. Bounding on structure instead of distance cannot.
        #
        # Both patterns tolerate a trailing comment. The first attempt
        # required `null` at end of line and failed on `sha256: null # PIN ME`
        # — which is exactly how an author marks the line they want filled in.
        lines = text.split("\n")
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if re.match(rf"\s*-?\s*name:\s*{re.escape(name)}\s*(#.*)?$", line)
            ),
            None,
        )
        if start is None:
            print(f"  x {name} — could not find its block in the YAML")
            return 1

        indent = len(lines[start]) - len(lines[start].lstrip())
        end = len(lines)
        for i in range(start + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            this_indent = len(lines[i]) - len(lines[i].lstrip())
            # The next list item, or anything dedented out of the block.
            if stripped.startswith("- ") or this_indent < indent:
                end = i
                break

        wrote_hash = False
        for i in range(start, end):
            # Matches an existing HASH as well as `null` — otherwise --force
            # computes the new digest and then silently fails to write it,
            # which is worse than not offering --force at all.
            if re.match(r"\s*sha256:\s*(null|~|[0-9a-fA-F]{64})?\s*(#.*)?$", lines[i]):
                pad = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                lines[i] = f"{pad}sha256: {digest}"
                wrote_hash = True
            # `bytes:` is measured, never estimated — fill it from the same
            # download rather than leaving the author to copy it by hand.
            elif re.match(r"\s*bytes:\s*[\d_]+\s*(#.*)?$", lines[i]):
                pad = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                lines[i] = f"{pad}bytes: {size} # exact, from the pinned download"

        if not wrote_hash:
            print(f"  x {name} — no `sha256:` line found in its block")
            return 1
        changed += 1
        text = "\n".join(lines)

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"\npinned {changed} asset(s) in {path.relative_to(ROOT)}")
    else:
        print("\nnothing to pin")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    if len(args) != 1:
        print(__doc__)
        sys.exit(2)
    sys.exit(pin(args[0], force=force))
