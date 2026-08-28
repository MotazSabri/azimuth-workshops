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
import time
import urllib.error
import urllib.request
from pathlib import Path

from .errors import fail

CHUNK = 1 << 16
TIMEOUT = 60
#: A truncated read is the ordinary failure on a domestic connection, not an
#: exceptional one — a few megabytes over a link that hiccups once. Retrying
#: costs seconds; failing the workshop costs the whole run.
ATTEMPTS = 3
BACKOFF = 2.0


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
        for attempt in range(1, ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                    data = response.read()
            # BARE Exception, deliberately. The first version caught
            # (URLError, HTTPError, TimeoutError, OSError) and a real download
            # died on http.client.IncompleteRead — which is an HTTPException,
            # NOT an OSError. It escaped as a raw traceback instead of the
            # bilingual AZ-E301 this module exists to produce.
            #
            # Enumerating network exception types is a losing game: every
            # layer adds its own, and the ones you miss are exactly the ones
            # that surface at 3am on someone else's connection. Anything that
            # goes wrong while fetching a URL is a fetch failure.
            except Exception as exc:
                attempts.append(f"{url} ({type(exc).__name__}, try {attempt})")
                if attempt < ATTEMPTS:
                    if not quiet:
                        print(f"  · {name} — {type(exc).__name__}, retrying ({attempt}/{ATTEMPTS})")
                    time.sleep(BACKOFF * attempt)
                    continue
                break

            # A truncated read can also arrive as a SHORT FILE with no error
            # at all, so the declared size is checked before the hash: it
            # names the problem ("got 3.1 of 3.9 MB") where a hash mismatch
            # would only say the bytes are wrong.
            expected_bytes = asset.get("bytes")
            if expected_bytes and len(data) != expected_bytes:
                attempts.append(f"{url} (short read: {len(data)} of {expected_bytes})")
                if attempt < ATTEMPTS:
                    if not quiet:
                        print(
                            f"  · {name} — short read "
                            f"({len(data):,} of {expected_bytes:,}), retrying "
                            f"({attempt}/{ATTEMPTS})"
                        )
                    time.sleep(BACKOFF * attempt)
                    continue
                break

            dest.write_bytes(data)

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
