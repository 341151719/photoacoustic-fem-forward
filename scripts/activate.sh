#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd -- "${PROJECT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  VENV_DIR="${PROJECT_DIR}/.venv"
else
  VENV_DIR="${PARENT_DIR}/.venv"
fi
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "warning: virtual environment not found at ${VENV_DIR}" >&2
fi
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
LOCAL_GLU="${PARENT_DIR}/.system-libs/root/usr/lib/x86_64-linux-gnu"
if [[ -d "${LOCAL_GLU}" ]]; then
  # Keep any system installation first; append the workspace fallback.
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${LOCAL_GLU}"
fi
echo "pa_fem environment: ${PROJECT_DIR}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  echo "using Python environment: ${VENV_DIR}"
fi
if [[ -d "${LOCAL_GLU}" ]]; then
  echo "local Gmsh runtime fallback available: ${LOCAL_GLU}"
else
  echo "libGLU not bundled; install libglu1-mesa or set PA_FEM_GLU_LIBRARY"
fi
