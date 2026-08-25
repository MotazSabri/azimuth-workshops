"""Fetch a declared asset, or fail with AZ-E301/E302 and a reason.

Datasets are the most perishable thing in a notebook corpus. A URL that worked
when the workshop was written is a URL that will 404 in two years, usually
silently — pandas will happily read an HTML error page as a one-column CSV and
the workshop will teach nonsense instead of failing.

Two defences, both cheap:

  * **Mirrors.** Every asset declares a list of sources, tried in order. The
    first is the canonical home; later ones are copies we control.
  * **A hash.** The bytes are checked before anything parses them. A file that
    downloaded but changed is AZ-E302 — a different failure from a file that
    would not download at all (AZ-E301), because they need different fixes:
    one is "try again", the other is "we need to update the pin".

`sha256: null` in the manifest means "not pinned yet" and only warns. That is
the honest state during authoring; the linter is what should refuse it at
publish time.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from pathlib import Path

from .errors import fail

CHUNK = 1 << 16
TIMEOUT = 60


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(asset: dict, dest_dir: Path, lang: str = "en", quiet: bool = False) -> Path:
    """Download one declared asset and return its local path.

    Idempotent: a file already present with the right hash is not fetched
    again, so re-running the cell after a disconnect costs nothing.
    """
    name = asset["name"]
    expected = asset.get("sha256")
    sources = asset.get("sources") or ([asset["url"]] if asset.get("url") else [])
    if not sources:
        raise fail("AZ-E301", lang=lang, asset=name)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    if dest.exists():
        if not expected or _sha256(dest) == expected:
            if not quiet:
                print(f"  · {name} — already present")
            return dest
        dest.unlink()  # present but wrong: refetch rather than trust it

    attempts: list[str] = []
    for url in sources:
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                data = response.read()
            dest.write_bytes(data)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            attempts.append(f"{url} ({type(exc).__name__})")
            continue

        if expected:
            actual = _sha256(dest)
            if actual != expected:
                dest.unlink(missing_ok=True)
                raise fail(
                    "AZ-E302",
                    lang=lang,
                    asset=name,
                    expected=expected[:12],
                    found=actual[:12],
                )
        if not quiet:
            size_kb = dest.stat().st_size / 1024
            pin = "verified" if expected else "UNPINNED"
            print(f"  · {name} — {size_kb:,.0f} KB, {pin}")
        return dest

    raise fail("AZ-E301", lang=lang, asset=name, tried=" | ".join(attempts))
