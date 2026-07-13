"""Self-supervised training entry points.

``train(...)`` runs (or resumes) AttMask pretraining, single-GPU-first
(CUDA / MPS / CPU), with seamless step-level resume and clean Ctrl-C
checkpointing. ``finetune(...)`` (Phase 7) branches from a base checkpoint.

Continual mode (replay) is Phase 7; ``mode="continual"`` currently trains like
``scratch`` on the given data (replay wiring lands with the lineage work).
"""

from __future__ import annotations

import datetime as _dt
import json
import signal
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mole.config import config_hash, load_config
from mole.progress import track
from mole.selfsup._train_utils import (cancel_gradients_last_layer, clip_gradients,
                                       cosine_scheduler)
from mole.selfsup.attmask import AttMask
from mole.selfsup.checkpoint import find_resume, load_checkpoint, save_checkpoint
from mole.selfsup.dataset import PatchWindowDataset
from mole.selfsup.head import iBOTHead
from mole.selfsup.loss import iBOTLoss
from mole.selfsup.vit import build_vit
from mole.selfsup.wrapper import MultiCropWrapper, get_params_groups

ATTMASK_MODES = ("attmask_high", "attmask_hint", "attmask_low")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ema_update(student: nn.Module, teacher: nn.Module, m: float) -> None:
    """EMA teacher update over the parameter NAMES common to both (teacher may
    share its projection last-layer, so it has fewer params than the student)."""
    teacher_params = dict(teacher.named_parameters())
    for name, p in student.named_parameters():
        if name in teacher_params:
            teacher_params[name].data.mul_(m).add_((1 - m) * p.detach().data)


def _rand_masks(batch: int, n_tokens: int, ratio: float, device) -> torch.Tensor:
    """Random per-sample token mask (for pred_shape='rand')."""
    k = int(round(ratio * n_tokens))
    m = torch.zeros(batch, n_tokens, dtype=torch.bool, device=device)
    if k > 0:
        for i in range(batch):
            idx = torch.randperm(n_tokens, device=device)[:k]
            m[i, idx] = True
    return m


def _build_loader(dataset, batch_size, num_workers, epoch, seed, pin_memory=False):
    g = torch.Generator()
    g.manual_seed(seed * 100003 + epoch)  # deterministic per-epoch order for resume
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


def train(config_path: str | Path, output_dir: str | Path | None = None,
          mode: Literal["scratch", "continual"] = "scratch",
          resume: str | Path | None = None, overrides: list[str] | None = None):
    """Run (or resume) AttMask pretraining. Auto-resumes if the run dir has a checkpoint."""
    import warnings

    # Cosmetic: we deliberately keep the classic weight_norm (checkpoint-format
    # continuity with the original AttMask heads).
    warnings.filterwarnings("ignore", message=r".*torch\.nn\.utils\.weight_norm.*deprecated.*")

    cfg = load_config(config_path, overrides)
    if output_dir:
        cfg["train"]["output_dir"] = str(output_dir)
    run_dir = Path(cfg["train"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["train"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

    device = _pick_device()
    use_fp16 = bool(cfg["train"]["fp16"]) and device.type == "cuda"
    print(f"[mole] device={device} fp16={use_fp16} mode={mode}")

    # ---- data ----
    d = cfg["data"]
    dataset = PatchWindowDataset(
        d["path"], window_size=d["window_size"], model_size=d["model_size"],
        overlap=d["overlap"], use_zones=d["use_zones"], preset=cfg["aug"]["preset"],
        pred_ratio=cfg["mask"]["pred_ratio"], pred_ratio_var=cfg["mask"]["pred_ratio_var"],
        pred_start_epoch=cfg["mask"]["pred_start_epoch"],
    )
    ac = dataset.aug_config
    ngc, nlc = ac.global_crops_number, ac.local_crops_number
    print(f"[mole] {dataset.n_images} images -> {len(dataset)} windows; crops {ngc} global + {nlc} local")

    # ---- models ----
    m = cfg["model"]
    student_backbone = build_vit(m["arch"], patch_size=m["patch_size"], drop_path_rate=m["drop_path"],
                                 return_all_tokens=True, masked_im_modeling=m["use_masked_im_modeling"],
                                 num_class_tokens=m["num_class_tokens"])
    teacher_backbone = build_vit(m["arch"], patch_size=m["patch_size"], return_all_tokens=True,
                                 num_class_tokens=m["num_class_tokens"])
    embed_dim = student_backbone.embed_dim
    student = MultiCropWrapper(student_backbone, iBOTHead(
        embed_dim, m["out_dim"], patch_out_dim=m["patch_out_dim"], norm=m["norm_in_head"],
        act=m["act_in_head"], norm_last_layer=m["norm_last_layer"], shared_head=m["shared_head"],
        num_class_tokens=m["num_class_tokens"])).to(device)
    teacher = MultiCropWrapper(teacher_backbone, iBOTHead(
        embed_dim, m["out_dim"], patch_out_dim=m["patch_out_dim"], norm=m["norm_in_head"],
        act=m["act_in_head"], shared_head=m["shared_head_teacher"],
        num_class_tokens=m["num_class_tokens"])).to(device)
    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- loss / optimizer ----
    same_dim = m["shared_head"] or m["shared_head_teacher"]
    lo = cfg["loss"]
    ibot_loss = iBOTLoss(m["out_dim"], m["out_dim"] if same_dim else m["patch_out_dim"], ngc, nlc,
                         lo["warmup_teacher_temp"], lo["teacher_temp"], lo["warmup_teacher_patch_temp"],
                         lo["teacher_patch_temp"], lo["warmup_teacher_temp_epochs"], cfg["train"]["epochs"],
                         lambda1=lo["lambda1"], lambda2=lo["lambda2"],
                         mim_start_epoch=cfg["mask"]["pred_start_epoch"]).to(device)

    o = cfg["optim"]
    optimizer = {"adamw": torch.optim.AdamW, "sgd": lambda pg: torch.optim.SGD(pg, lr=0, momentum=0.9)}\
        .get(o["optimizer"], torch.optim.AdamW)(get_params_groups(student))
    fp16_scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = len(dataset) // o["batch_size"]
    if steps_per_epoch == 0:
        raise ValueError(f"batch_size {o['batch_size']} > windows {len(dataset)}; lower optim.batch_size.")
    lr_peak = o["lr"] * o["batch_size"] / 256.0
    lr_sched = cosine_scheduler(lr_peak, o["min_lr"], epochs, steps_per_epoch, o["warmup_epochs"])
    wd_sched = cosine_scheduler(o["weight_decay"], o["weight_decay_end"], epochs, steps_per_epoch)
    mom_sched = cosine_scheduler(m["momentum_teacher"], 1, epochs, steps_per_epoch)

    # ---- resume ----
    resume_path = Path(resume) if resume else find_resume(run_dir)
    start_step = 0
    if resume_path and Path(resume_path).is_file():
        start_step = load_checkpoint(resume_path, student=student, teacher=teacher, optimizer=optimizer,
                                     ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, map_location=device)
        print(f"[mole] resumed from {resume_path} at step {start_step}")

    _write_manifest(run_dir, cfg, dataset, mode, resume_path, start_step)

    # ---- Ctrl-C: checkpoint then exit cleanly ----
    state = {"stop": False}
    signal.signal(signal.SIGINT, lambda *_: state.update(stop=True))

    n_tokens = (d["model_size"] // m["patch_size"]) ** 2
    nct = m["num_class_tokens"]
    pred_shape = cfg["mask"]["pred_shape"]
    start_epoch, skip = divmod(start_step, steps_per_epoch)
    it = start_step
    print(f"[mole] training {epochs} epochs × {steps_per_epoch} steps (start epoch {start_epoch}, step {it})")

    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        loader = _build_loader(dataset, o["batch_size"], d["num_workers"], epoch, seed,
                               pin_memory=(device.type == "cuda"))
        bar = track(loader, f"epoch {epoch + 1}/{epochs}", total=steps_per_epoch, unit="step")
        for i, (images, _) in enumerate(bar):
            if epoch == start_epoch and i < skip:
                continue
            for j, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_sched[it]
                if j == 0:
                    pg["weight_decay"] = wd_sched[it]
            images = [im.to(device, non_blocking=True) for im in images]
            globals_, locals_ = images[:ngc], images[ngc:]

            with torch.autocast(device_type=device.type, enabled=use_fp16):
                if pred_shape in ATTMASK_MODES:
                    teacher_output, teacher_attn = teacher(globals_, return_attention=True)
                    cls_attn = teacher_attn[:, :, :nct, nct:].mean((1, 2)).detach()
                    pr = dataset.get_pred_ratio()
                    masks = AttMask(cls_attn, cfg["mask"]["masking_prob"], pred_shape, pr,
                                    cfg["mask"]["show_max"] * pr, cfg["mask"]["show_max"])
                    masks = list(masks.chunk(ngc, 0))
                elif pred_shape == "rand":
                    teacher_output = teacher(globals_)
                    pr = dataset.get_pred_ratio()
                    masks = list(_rand_masks(globals_[0].shape[0] * ngc, n_tokens, pr, device).chunk(ngc, 0))
                else:
                    raise NotImplementedError(f"pred_shape={pred_shape!r} not supported (use attmask_*/rand)")

                student_output = student(globals_, mask=masks)
                student.backbone.masked_im_modeling = False
                student_local_cls = student(locals_)[0] if locals_ else None
                student.backbone.masked_im_modeling = m["use_masked_im_modeling"]

                all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
                loss = all_loss.pop("loss")

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is {loss.item()} at step {it}; stopping.")

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                if o["clip_grad"]:
                    clip_gradients(student, o["clip_grad"])
                cancel_gradients_last_layer(epoch, student, o["freeze_last_layer"])
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if o["clip_grad"]:
                    fp16_scaler.unscale_(optimizer)
                    clip_gradients(student, o["clip_grad"])
                cancel_gradients_last_layer(epoch, student, o["freeze_last_layer"])
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            with torch.no_grad():
                _ema_update(student, teacher, mom_sched[it])

            it += 1
            bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_sched[it - 1]:.2e}")

            if it % int(cfg["train"]["save_every_steps"]) == 0:
                save_checkpoint(run_dir, student=student, teacher=teacher, optimizer=optimizer,
                                ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, global_step=it, config=cfg)
            if state["stop"]:
                save_checkpoint(run_dir, student=student, teacher=teacher, optimizer=optimizer,
                                ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, global_step=it, config=cfg)
                print(f"\n[mole] interrupted — checkpointed at step {it} → {run_dir}/checkpoint.pth")
                return

        snap = epoch if (cfg["train"]["saveckp_epoch_freq"] and epoch
                         and epoch % cfg["train"]["saveckp_epoch_freq"] == 0) else None
        save_checkpoint(run_dir, student=student, teacher=teacher, optimizer=optimizer,
                        ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, global_step=it, config=cfg,
                        epoch_snapshot=snap)
        with (run_dir / "log.txt").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "step": it, "loss": float(loss.item())}) + "\n")

    print(f"[mole] done — {it} steps → {run_dir}/checkpoint.pth")
    return run_dir


def _write_manifest(run_dir, cfg, dataset, mode, resume_path, start_step):
    manifest = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "config_hash": config_hash(cfg),
        "parent_checkpoint": str(resume_path) if resume_path else None,
        "resumed_at_step": start_step,
        "seed": cfg["train"]["seed"],
        "arch": cfg["model"]["arch"],
        "dataset_root": cfg["data"]["path"],
        "n_images": dataset.n_images,
        "n_windows": len(dataset),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")


def finetune(config_path: str | Path, base_checkpoint: str | Path,
             output_dir: str | Path | None = None, overrides: list[str] | None = None):
    """Branch a dataset-specific finetune from a base checkpoint (Phase 7)."""
    raise NotImplementedError("Finetuning is implemented in Phase 7.")
