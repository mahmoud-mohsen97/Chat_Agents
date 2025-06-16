"""Researcher Agent Module

A minimal, self-contained agent that performs research using a Task -> Planner -> Researcher -> Publisher flow.
"""

from .graph import build_graph

__all__ = ["build_graph"] 