from __future__ import annotations

import torch

from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)


def pick_device(preferred: str | None = None) -> torch.device:
    """Pick a torch device with priority: explicit > CUDA > MPS > CPU.

    Logs the choice and the reason so runs are self-documenting.
    """
    if preferred is not None and preferred != "auto":
        device = torch.device(preferred)
        _log.info("device.selected name=%s reason=explicit", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _log.info("device.selected name=%s reason=cuda_available count=%d",
                  device, torch.cuda.device_count())
        return device

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        device = torch.device("mps")
        _log.info("device.selected name=%s reason=mps_available", device)
        return device

    device = torch.device("cpu")
    _log.info("device.selected name=%s reason=fallback", device)
    return device
