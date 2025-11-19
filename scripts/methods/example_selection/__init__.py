"""
Example selection utilities for disguise methods.

This module re-exports selectors from methods.utils to provide a clean, stable
import path (scripts.methods.example_selection) without breaking existing utils imports.
"""
try:
    from ..utils.adaptive_example_selector import AdaptiveExampleSelector  # noqa: F401
except Exception:  # pragma: no cover
    AdaptiveExampleSelector = None  # type: ignore
