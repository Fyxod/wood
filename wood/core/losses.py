"""WOOD loss helpers."""
from __future__ import annotations


def wood_loss(Z):
    """The WOOD scalar objective is optimized through loss = -Z."""

    return -Z
