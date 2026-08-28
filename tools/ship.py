#!/usr/bin/env python3
"""One command that runs the whole pipeline for a workshop, in order.

    python tools/ship.py <slug>            # validate, build, execute, report
    python tools/ship.py <slug> --check    # everything except executing
    python tools/ship.py --all

WHY THIS EXISTS

The steps are: validate, build notebooks, fetch assets, execute, check drift,
commit the right directories, push, promote, bump the submodule, rebuild the
site. Ten of them, order-dependent, and getting one wrong fails somewhere
else entirely — a skipped `git add` shows up as "1 workshop" on the site, a
skipped rebuild shows up as a codeHash mismatch on a page.

Nothing here is new. It runs the same tools in the only order that works, and
STOPS at the first failure instead of letting a later step report a confusing
symptom of an earlier one. What it does not do is anything involving git or
the site: those are decisions, and they stay yours.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, args: list[str]) -> bool:
    print(f"\n── {label} " + "─" * max(0, 58 - len(label)))
    result = subprocess.run([sys.executable, *args], cwd=ROOT, check=False)
    return result.returncode == 0


def status(slug: str) -> None:
    """What the SITE will make of this workshop, said in the workshops repo.

    The site is the only place this was visible before, which meant a full
    submodule bump and content rebuild just to find out whether a run had
    been captured.
    """
    import platform as _platform

    import yaml

    spec = yaml.safe_load((ROOT / "workshops" / slug / "workshop.yaml").read_text(encoding="utf-8"))
    print(f"\n── {slug} " + "─" * max(0, 58 - len(slug)))

    # Say it BEFORE the run status, not after: if this machine cannot execute
    # the workshop, "not publishable yet" is true but misleading — nothing is
    # wrong with the workshop.
    default = next((p for p in spec.get("profiles", []) if p.get("default")), None)
    platforms = ((default or {}).get("requires") or {}).get("platforms")
    if platforms and _platform.system() not in platforms:
        print(
            f"   NOTE        this workshop declares platforms {platforms} and you are on "
            f"{_platform.system()}.\n"
            f"               it cannot be executed here — run it on Colab or a Linux box,\n"
            f"               or let the GPU CI workflow capture the run."
        )
    print(f"   status      {spec.get('status')}   published: {spec.get('publishedAt') or '—'}")

    for lang in ("en", "ar"):
        path = ROOT / "generated" / "runs" / slug / f"{lang}.manifest.json"
        if not path.exists():
            print(f"   run ({lang})    no manifest — the page will say 'outputs pending'")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"   run ({lang})    {manifest.get('status')} · "
            f"{len(manifest.get('metrics') or {})} metric(s) · "
            f"codeHash {manifest.get('codeHash')}"
        )

    # The notebook a reader opens comes from the `stable` BRANCH, not from
    # the working tree. If stable is behind, the Colab badge serves an old
    # workshop while everything here looks current — which is exactly how
    # three rounds of results came back from a superseded code.py.
    try:
        ahead = subprocess.run(
            ["git", "rev-list", "--count", "stable..HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if ahead.returncode == 0 and ahead.stdout.strip().isdigit():
            n = int(ahead.stdout.strip())
            if n:
                print(
                    f"   branch      stable is {n} commit(s) BEHIND — the Colab badge "
                    f"still serves the old notebook"
                )
    except OSError:
        pass

    ready = spec.get("status") == "stable" and spec.get("publishedAt")
    verified = all(
        (ROOT / "generated" / "runs" / slug / f"{lang}.manifest.json").exists()
        and json.loads(
            (ROOT / "generated" / "runs" / slug / f"{lang}.manifest.json").read_text(
                encoding="utf-8"
            )
        ).get("status")
        == "verified"
        for lang in ("en", "ar")
    )
    if ready and verified:
        print("   -> ready to publish")
    elif verified:
        print("   -> runs verified; set status: stable + publishedAt to publish")
    else:
        print("   -> not publishable yet: both languages need a verified run")


def ship(slug: str, check_only: bool) -> int:
    if not run("validate", ["tools/validate.py"]):
        print("\nstopped: fix the errors above before anything else.")
        return 1
    if not run("build notebooks", ["tools/build_notebooks.py"]):
        return 1
    if not check_only and not run("execute", ["tools/execute.py", slug]):
        print("\nexecution did not verify — the first failure is printed above.")
        status(slug)
        return 1
    status(slug)

    print(
        "\nnext, if the above says ready:\n"
        f"   git add workshops/{slug} generated/notebooks generated/runs\n"
        '   git commit -m "workshop: ' + slug + '"\n'
        "   git push origin main && git push origin main:stable\n"
        "\nthen in the SITE repo:\n"
        "   git -C vendor/azimuth-workshops fetch\n"
        "   git -C vendor/azimuth-workshops checkout origin/stable\n"
        "   npm run content     # expect this workshop to show 'verified'\n"
        "   npm run build\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true", help="do not execute")
    args = parser.parse_args()

    slugs = (
        sorted(d.name for d in (ROOT / "workshops").iterdir() if d.is_dir())
        if args.all
        else [args.slug]
        if args.slug
        else []
    )
    if not slugs:
        print(__doc__)
        return 2

    worst = 0
    for slug in slugs:
        worst = max(worst, ship(slug, args.check))
    return worst


if __name__ == "__main__":
    sys.exit(main())
