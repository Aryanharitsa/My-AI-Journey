"""Trainer for from-scratch retrieval encoders (Session 03 / Phase 5).

One architecture-agnostic trainer. Takes any ``Encoder`` that exposes a
``.body`` (nn.Module that returns normalized-ready embeddings from
``(input_ids, attention_mask)``) and a ``.tokenizer`` (HF tokenizer).

Training loop specifics (handoff §5.5):
    - InfoNCE loss with in-batch negatives, tau=0.05
    - AdamW(lr=1e-4, wd=0.01)
    - Linear warmup over first 10% of steps, cosine decay after
    - AMP (autocast + GradScaler) optional; disabled with --no-amp
    - Checkpoint best-val-loss + final-step to ``output_dir/<encoder>/``
    - Validation every 500 steps on a held-out triplet set
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vitruvius.training.contrastive import InfoNCELoss
from vitruvius.utils.logging import get_logger
from vitruvius.utils.seed import set_seed

_log = get_logger(__name__)


@dataclass
class TrainConfig:
    encoder: str
    train_path: str
    val_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 64
    max_seq_len: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.10
    temperature: float = 0.05
    amp: bool = True
    val_every: int = 500
    seed: int = 1729
    log_every: int = 50


class TripletDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        self._rows: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        r = self._rows[idx]
        return {"query": r["query"], "positive": r["positive"], "negative": r["negative"]}


def _make_collate(tokenizer, max_len: int):
    def collate(batch: list[dict]) -> dict:
        qs = [b["query"] for b in batch]
        ps = [b["positive"] for b in batch]
        ns = [b["negative"] for b in batch]
        qe = tokenizer(qs, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        pe = tokenizer(ps + ns, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return {
            "q_input_ids": qe["input_ids"],
            "q_attn": qe["attention_mask"],
            "p_input_ids": pe["input_ids"],
            "p_attn": pe["attention_mask"],
            "batch_size": len(batch),
        }
    return collate


def _linear_warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _eval_loss(
    body: nn.Module,
    loader: DataLoader,
    loss_fn: InfoNCELoss,
    device: torch.device,
    amp: bool,
    max_batches: int = 50,
) -> float:
    body.eval()
    losses = []
    autocast_ctx = torch.cuda.amp.autocast if amp and device.type == "cuda" else _null_autocast
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            q_ids = batch["q_input_ids"].to(device)
            q_attn = batch["q_attn"].to(device)
            p_ids = batch["p_input_ids"].to(device)
            p_attn = batch["p_attn"].to(device)
            with autocast_ctx():
                q_out = nn.functional.normalize(body(q_ids, q_attn), dim=-1)
                p_out = nn.functional.normalize(body(p_ids, p_attn), dim=-1)
                loss = loss_fn(q_out, p_out)
            losses.append(float(loss.item()))
    body.train()
    return sum(losses) / max(1, len(losses))


class _null_autocast:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def train(cfg: TrainConfig, body: nn.Module, tokenizer) -> dict:
    """Run one training pass. Returns a summary dict suitable for serialization."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("train.start cfg=%s device=%s", cfg.encoder, device)

    body.to(device).train()

    train_ds = TripletDataset(cfg.train_path)
    val_ds = TripletDataset(cfg.val_path)
    collate = _make_collate(tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
        collate_fn=collate, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2,
        collate_fn=collate, drop_last=False,
    )

    n_params = sum(p.numel() for p in body.parameters())
    total_steps = (len(train_ds) // cfg.batch_size) * cfg.epochs
    warmup_steps = int(cfg.warmup_frac * total_steps)
    _log.info(
        "train.schedule params=%.2fM steps=%d warmup=%d batch=%d epochs=%d",
        n_params / 1e6, total_steps, warmup_steps, cfg.batch_size, cfg.epochs,
    )

    optim = torch.optim.AdamW(body.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    loss_fn = InfoNCELoss(temperature=cfg.temperature)

    autocast_ctx = torch.cuda.amp.autocast if (cfg.amp and device.type == "cuda") else _null_autocast
    curve: list[dict] = []
    best_val = float("inf")
    out_dir = Path(cfg.output_dir) / cfg.encoder
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t_start = time.perf_counter()
    peak_mem_mb = 0
    try:
        for epoch in range(cfg.epochs):
            for batch in train_loader:
                step += 1
                lr_scale = _linear_warmup_cosine(step, total_steps, warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = cfg.lr * lr_scale

                q_ids = batch["q_input_ids"].to(device, non_blocking=True)
                q_attn = batch["q_attn"].to(device, non_blocking=True)
                p_ids = batch["p_input_ids"].to(device, non_blocking=True)
                p_attn = batch["p_attn"].to(device, non_blocking=True)

                with autocast_ctx():
                    q_out = nn.functional.normalize(body(q_ids, q_attn), dim=-1)
                    p_out = nn.functional.normalize(body(p_ids, p_attn), dim=-1)
                    loss = loss_fn(q_out, p_out)

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(body.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()

                if device.type == "cuda":
                    peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))

                if step % cfg.log_every == 0:
                    _log.info(
                        "train.step step=%d/%d epoch=%d loss=%.4f lr=%.2e",
                        step, total_steps, epoch, float(loss.item()),
                        cfg.lr * lr_scale,
                    )
                    curve.append({"step": step, "train_loss": float(loss.item()),
                                  "val_loss": None})

                if step % cfg.val_every == 0:
                    val = _eval_loss(body, val_loader, loss_fn, device, cfg.amp)
                    _log.info("train.val step=%d val_loss=%.4f", step, val)
                    curve.append({"step": step, "train_loss": float(loss.item()),
                                  "val_loss": val})
                    if val < best_val:
                        best_val = val
                        torch.save(
                            {"body_state_dict": body.state_dict(), "step": step, "val_loss": val},
                            out_dir / "best.pt",
                        )
                        _log.info("train.checkpoint.best step=%d val_loss=%.4f", step, val)
    except KeyboardInterrupt:
        _log.warning("train.interrupted step=%d", step)

    # Final checkpoint
    final_val = _eval_loss(body, val_loader, loss_fn, device, cfg.amp)
    torch.save(
        {"body_state_dict": body.state_dict(), "step": step, "val_loss": final_val},
        out_dir / "final.pt",
    )
    wall = time.perf_counter() - t_start

    summary = {
        "encoder": cfg.encoder,
        "config": asdict(cfg),
        "steps_completed": step,
        "total_steps_planned": total_steps,
        "best_val_loss": best_val if best_val != float("inf") else None,
        "final_val_loss": final_val,
        "final_train_loss": curve[-1]["train_loss"] if curve else None,
        "wall_seconds": round(wall, 2),
        "param_count": n_params,
        "peak_gpu_memory_mb": int(peak_mem_mb),
        "device": str(device),
        "amp_enabled": bool(cfg.amp and device.type == "cuda"),
    }
    return summary, curve
