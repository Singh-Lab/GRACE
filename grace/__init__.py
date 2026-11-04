"""GRACE: gradient-resolved adaptive constrained exploration for sequence redesign.

Public entry points:

    from grace import optimize, list_tasks, load_task

`optimize` runs one constrained redesign and returns a Design. The command-line
equivalent is `python scripts/optimize.py`, documented in the README.
"""
from .api import Design, optimize           # noqa: F401
from .registry import list_tasks, load_task  # noqa: F401

__version__ = "1.0.0"
__all__ = ["Design", "optimize", "list_tasks", "load_task"]
