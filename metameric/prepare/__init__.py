"""Prepare word lists for analysis using metameric."""
from .data import process_data, process_and_write
from .weights import IA_WEIGHTS


__all__ = ["process_data",
           "process_and_write",
           "IA_WEIGHTS"]
