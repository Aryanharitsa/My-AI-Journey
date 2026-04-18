"""Pytest setup: apply the macOS libomp workaround before any test imports."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
