"""Torch Inductor logical-buffer tracing, comparison, and visualization tools."""

from .logic_monkeypatch import install, maybe_auto_viz

__all__ = ["install", "maybe_auto_viz"]
