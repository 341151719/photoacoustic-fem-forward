"""Independent two-dimensional photoacoustic FEM forward model.

The package contains only the physical forward model.  It deliberately does
not implement random illumination, compressed sensing, or image
reconstruction.
"""

__version__ = "0.1.0"

from .config import CaseConfig, load_config

__all__ = ["CaseConfig", "load_config", "__version__"]
