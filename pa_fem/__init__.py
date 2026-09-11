"""Independent two-dimensional photoacoustic FEM forward model.

The package contains the physical forward model and an optional idealized
dual-encoded data-generation prototype. It deliberately does not implement
compressed-sensing image reconstruction.
"""

__version__ = "0.2.0"

from .config import CaseConfig, load_config

__all__ = ["CaseConfig", "load_config", "__version__"]
