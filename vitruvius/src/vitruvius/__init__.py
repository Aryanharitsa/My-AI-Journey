"""Project Vitruvius — dense retrieval encoder architecture study."""

import os as _os

# macOS workaround: faiss-cpu and torch each ship their own libomp; loading
# both crashes with OMP error #15. No-op on Linux pods (faiss-gpu links
# differently). Setting via setdefault means an explicit operator override wins.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.8.0"
