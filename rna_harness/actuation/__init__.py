"""Actuation-fidelity instrumentation + structured folding-mismatch diagnosis.

Operationalizes the two paper axes: diagnosis() gives the T_trace signal (exact per-
position credit for what folded wrong), and instrument.py measures A (whether that
diagnosis actually controlled the next action).
"""
from __future__ import annotations

from .diagnosis import Diagnosis, diagnose  # noqa: F401

__all__ = ["Diagnosis", "diagnose"]
