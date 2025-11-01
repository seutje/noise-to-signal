"""
Renderer package for the noise-to-signal offline rendering pipeline.

Modules are structured to separate configuration loading, audio feature
extraction, latent controller logic, decoder integration, and output
orchestration. See `render_album.py` for the primary CLI.
"""

from __future__ import annotations

from pathlib import Path

# Base directory convenient for locating bundled presets/templates.
PACKAGE_ROOT = Path(__file__).resolve().parent

__all__ = ["PACKAGE_ROOT"]

