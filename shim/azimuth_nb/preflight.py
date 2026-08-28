"""Check the runtime before the learner spends twenty minutes finding out.

The failure this exists to prevent: someone opens the notebook, runs every
cell, waits through a download and eight epochs, and the training cell dies on
an out-of-memory error at minute nineteen. Everything needed to predict that
was knowable in the first two seconds.

So preflight runs once, at setup, and refuses early and legibly.

TORCH IS ASSERTED, NEVER INSTALLED. Colab ships a PyTorch built against its
own driver; `pip install torch` on top of it fetches a wheel that may not
match, and the resulting CUDA errors name nothing that would lead a learner
back to the pip line that caused them. If torch is missing or too old, the
right answer is a fresh runtime, and AZ-E201/AZ-E202 say exactly that.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

from .errors import fail

#: Oldest PyTorch these workshops are written against. Below this, `torch.compile`
#: and several `nn` defaults differ enough that the prose would be wrong.
MIN_TORCH = (2, 0)


@dataclass
class Runtime:
    """What we found, in the units the profile is written in."""

    accelerator: str  # 'cuda' | 'mps' | 'cpu'
    device_name: str
    vram_gb: float  # 0.0 when there is no accelerator
    ram_gb: float | None  # None when the platform will not tell us
    disk_gb: float
    torch_version: str

    def summary(self, lang: str = "en") -> str:
        if lang == "ar":
            gpu = (
                f"{self.device_name} · {self.vram_gb:.1f} غ.ب"
                if self.accelerator == "cuda"
                else "بدون معالج رسوميات"
            )
            ram = f"ذاكرة {self.ram_gb:.1f} غ.ب" if self.ram_gb else "الذاكرة غير معروفة"
            return f"{gpu} · {ram} · PyTorch {self.torch_version}"
        gpu = (
            f"{self.device_name} · {self.vram_gb:.1f} GB"
            if self.accelerator == "cuda"
            else "no GPU"
        )
        ram = f"{self.ram_gb:.1f} GB RAM" if self.ram_gb else "RAM unknown"
        return f"{gpu} · {ram} · PyTorch {self.torch_version}"


def _ram_gb() -> float | None:
    """Total system RAM, or None when we cannot tell.

    os.sysconf exists on Linux and macOS — which covers every Colab runtime —
    but not on Windows, where it raises AttributeError. The first Windows run
    of this shim printed "0.0 GB RAM", which is not a degraded reading, it is
    a WRONG one: it says the machine has no memory. None is the honest answer,
    and the RAM check skips rather than comparing against a number it invented.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows, and a more accurate reading anywhere it exists
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return None


def inspect_runtime(lang: str = "en") -> Runtime:
    """Look at the machine. Raises AZ-E201/E202 if torch is unusable."""
    try:
        import torch
    except ImportError as exc:
        raise fail("AZ-E201", lang=lang) from exc

    parts = torch.__version__.split(".")
    try:
        major_minor = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        major_minor = (0, 0)
    if major_minor < MIN_TORCH:
        raise fail(
            "AZ-E202",
            lang=lang,
            found=torch.__version__,
            needed=f"{MIN_TORCH[0]}.{MIN_TORCH[1]}+",
        )

    if torch.cuda.is_available():
        accelerator = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Someone running this locally on a Mac. Not a target, but no reason
        # to refuse it when the workshop is small enough to fit.
        accelerator, device_name, vram_gb = "mps", "Apple GPU", 0.0
    else:
        accelerator, device_name, vram_gb = "cpu", "CPU", 0.0

    return Runtime(
        accelerator=accelerator,
        device_name=device_name,
        vram_gb=vram_gb,
        ram_gb=_ram_gb(),
        disk_gb=shutil.disk_usage("/").free / (1024**3),
        torch_version=torch.__version__,
    )


def check_profile(runtime: Runtime, profile: dict, lang: str = "en") -> None:
    """Hold the runtime against what the active profile declared it needs.

    `requiresGpu: false` is honoured: a workshop that runs fine on CPU should
    not refuse a CPU. That matters more than it sounds — it is what lets the
    site's own CI execute a workshop without a GPU runner when the workshop
    genuinely does not need one.
    """
    needs = profile.get("requires", {}) or {}

    # PLATFORM FIRST, because it is the cheapest check and the one whose
    # failure is otherwise least legible: a Linux-only CUDA kernel surfaces as
    # a bare WinError deep inside a model load, long after a download.
    platforms = needs.get("platforms")
    if platforms:
        current = (
            "linux"
            if sys.platform.startswith("linux")
            else "macos"
            if sys.platform == "darwin"
            else "windows"
            if sys.platform.startswith("win")
            else sys.platform
        )
        # Case-insensitive: the field is hand-written YAML, and `[Linux]` vs
        # `[linux]` is not a distinction anyone should have to get right.
        # An earlier version compared platform.system() ("Linux") against a
        # lowercase list and refused EVERY platform, including the correct one.
        if current not in {str(p).strip().lower() for p in platforms}:
            raise fail(
                "AZ-E104",
                lang=lang,
                needs=" or ".join(platforms),
                found=current,
            )

    if needs.get("gpu", True) and runtime.accelerator == "cpu":
        raise fail("AZ-E101", lang=lang)

    min_vram = float(needs.get("vramGb", 0) or 0)
    if min_vram and runtime.accelerator == "cuda" and runtime.vram_gb < min_vram:
        raise fail(
            "AZ-E102",
            lang=lang,
            needs=f"{min_vram:.0f} GB",
            found=f"{runtime.vram_gb:.1f} GB",
        )

    min_ram = float(needs.get("ramGb", 0) or 0)
    if min_ram and runtime.ram_gb and runtime.ram_gb < min_ram:
        raise fail(
            "AZ-E103",
            lang=lang,
            needs=f"{min_ram:.0f} GB",
            found=f"{runtime.ram_gb:.1f} GB",
        )
