"""Prepare word lists for analysis using diploria."""
from .data import process_data, process_and_write
from .weights import weights_to_matrix, IA_WEIGHTS


__all__ = ["process_data",
           "process_and_write",
           "weights_to_matrix",
           "IA_WEIGHTS"]
