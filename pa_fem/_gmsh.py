"""Load the Gmsh Python API with an explicit, auditable runtime check.

The manylinux Gmsh wheel needs ``libGLU.so.1`` even when the API is used in
headless mode.  A parent workspace may contain a locally extracted fallback
in ``光声/.system-libs``.  We preload that library when present; otherwise we
raise an actionable error instead of silently switching to a non-Gmsh mesh.
"""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from types import ModuleType


def _library_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    repository_root = here.parents[1]
    workspace_root = here.parents[2]
    candidates = [
        repository_root / ".system-libs/root/usr/lib/x86_64-linux-gnu/libGLU.so.1",
        repository_root / ".system-libs/usr/lib/x86_64-linux-gnu/libGLU.so.1",
        workspace_root / ".system-libs/root/usr/lib/x86_64-linux-gnu/libGLU.so.1",
        workspace_root / ".system-libs/usr/lib/x86_64-linux-gnu/libGLU.so.1",
    ]
    env_path = os.environ.get("PA_FEM_GLU_LIBRARY")
    if env_path:
        candidates.insert(0, Path(env_path))
    return candidates


def import_gmsh() -> ModuleType:
    """Import and return the real :mod:`gmsh` module.

    ``LD_LIBRARY_PATH`` is intentionally not modified globally.  Loading the
    locally supplied GLU shared object with ``RTLD_GLOBAL`` is sufficient for
    the Gmsh extension and keeps subprocess behaviour predictable.
    """

    errors: list[str] = []
    # Prefer a normal system installation.  Only if the dynamic linker cannot
    # find it do we try the workspace-local fallback.
    try:
        ctypes.CDLL("libGLU.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:  # pragma: no cover - platform-specific
        errors.append(f"system libGLU.so.1: {exc}")
        for candidate in _library_candidates():
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError as local_exc:  # pragma: no cover - platform-specific
                    errors.append(f"{candidate}: {local_exc}")
    try:
        return importlib.import_module("gmsh")
    except OSError as exc:
        hint = (
            "Gmsh Python API could not load its OpenGL utility dependency. "
            "Install libglu1-mesa (Linux) or set PA_FEM_GLU_LIBRARY to a "
            "compatible libGLU.so.1."
        )
        if errors:
            hint += " Preload attempts: " + "; ".join(errors)
        raise RuntimeError(hint) from exc
