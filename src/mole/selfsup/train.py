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
from mole.progress import progress_bar
from mole.selfsup._train_utils import (cancel_gradients_last_layer, clip_gradients,
                                       cosine_scheduler)
from mole.selfsup.attmask import AttMask
from mole.selfsup.checkpoint import (ARCH_FIELDS, filtered_load, find_resume,
                                     load_checkpoint, normalize_checkpoint, save_checkpoint)
from mole.selfsup.dataset import PatchWindowDataset
from mole.selfsup.head import iBOTHead
from mole.selfsup.loss import iBOTLoss
from mole.selfsup.vit import build_vit
from mole.selfsup.wrapper import MultiCropWrapper, get_params_groups

ATTMASK_MODES = ("attmask_high", "attmask_hint", "attmask_low")


def _make_tb_writer(run_dir, enabled: bool):
    """A TensorBoard SummaryWriter into the run dir, or None (never fatal).

    Event files land directly in the run dir, so `tensorboard --logdir runs` lists
    each run by its directory name. Missing tensorboard degrades to no-op logging.
    """
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:  # tensorboard not installed / import error — keep training
        print(f"[mole] TensorBoard logging off ({e}); `pip install tensorboard` to enable.")
        return None
    print(f"[mole] TensorBoard: writing scalars to {run_dir} "
          f"(view with `tensorboard --logdir {Path(run_dir).parent}`)")
    return SummaryWriter(log_dir=str(run_dir))


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


def _prior_init_from(run_dir) -> str | None:
    """The ``init_from`` recorded in a run dir's manifest, if any (for re-run detection)."""
    mpath = Path(run_dir) / "manifest.json"
    if not mpath.is_file():
        return None
    try:
        return json.loads(mpath.read_text()).get("init_from")
    except (json.JSONDecodeError, OSError):
        return None


def _adopt_arch(cfg: dict, src_cfg: dict, *, source: str) -> None:
    """Adopt the source checkpoint's architecture into ``cfg`` so its weights load.

    The weights only fit a matching model, so the checkpoint is authoritative on
    architecture; a conflicting user setting is overridden with a warning. Training
    hyperparameters (lr, epochs, data, aug, mask) stay from the user's config.
    """
    adopted = {}
    for f in ARCH_FIELDS:
        new = src_cfg.get("model", {}).get(f)
        if new is None:
            continue
        old = cfg["model"].get(f)
        if old != new:
            print(f"[mole] init-from: model.{f} {old!r} -> {new!r} (to match {source})")
        cfg["model"][f] = new
        adopted[f] = new
    print(f"[mole] init-from: adopted architecture {adopted}")


def _apply_warmstart(student, teacher, warm: dict) -> None:
    """Load foreign/mole weights into a fresh student & teacher (weights only, step 0)."""
    student_sd = warm["student"] if warm["student"] is not None else warm["teacher"]
    for module, sd, label in ((student, student_sd, "student"), (teacher, warm["teacher"], "teacher")):
        r = filtered_load(module, sd, label=label)
        note = ""
        if r["shape_mismatch"]:
            note += f", {len(r['shape_mismatch'])} shape-mismatch (re-init)"
        if r["missing"]:
            note += f", {len(r['missing'])} missing (re-init)"
        print(f"[mole] init-from: {label} loaded {r['loaded']}/{r['total']} params{note}")


def train(config_path: str | Path, output_dir: str | Path | None = None,
          mode: Literal["scratch", "continual"] = "scratch",
          resume: str | Path | None = None, overrides: list[str] | None = None,
          init_from: str | Path | None = None):
    """Run (or resume) AttMask pretraining. Auto-resumes if the run dir has a checkpoint.

    ``init_from`` warm-starts a fresh run from a foreign (original AttMask/iBOT) or
    mole checkpoint: it loads the weights only (not optimizer/RNG), adopts the
    source's architecture, and starts at step 0 with this config. Ignored if the
    run is resuming (a run dir with a checkpoint, or an explicit ``resume``).
    """
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

    # ---- warm-start peek (adopt source architecture before building the model) ----
    resume_ckpt = Path(resume) if (resume and Path(resume).is_file()) else find_resume(run_dir)
    will_resume = resume_ckpt is not None
    warm = None
    if init_from and will_resume:
        # A run dir with a checkpoint auto-resumes, which would ignore --init-from.
        # Allow it only for an idempotent re-run (this dir was itself started from the
        # SAME --init-from); otherwise the dir is stale/foreign — stop, don't silently
        # resume the wrong model.
        if _prior_init_from(run_dir) == str(init_from):
            print(f"[mole] resuming a run previously warm-started from {init_from}")
        else:
            print(f"[mole] ERROR: --init-from was given, but {run_dir} already holds a "
                  f"checkpoint ({resume_ckpt}) that was NOT started from {init_from}.\n"
                  f"        --init-from only *starts* a fresh run — it will not overwrite an "
                  f"existing one, and resuming the existing one would ignore your checkpoint.\n"
                  f"        • to warm-start from {init_from}: use a new --output-dir "
                  f"(or delete {resume_ckpt})\n"
                  f"        • to continue the run already in {run_dir}: drop --init-from "
                  f"(it auto-resumes)")
            raise SystemExit(1)
    elif init_from:
        raw = torch.load(init_from, map_location="cpu", weights_only=False)
        warm = normalize_checkpoint(raw)
        kind = "foreign" if warm["foreign"] else "mole"
        print(f"[mole] init-from ({kind}): warm-starting weights from {init_from}")
        _adopt_arch(cfg, warm["config"], source=str(init_from))

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

    # ---- resume / warm-start ----
    resume_path = resume_ckpt
    start_step = 0
    if resume_path and Path(resume_path).is_file():
        start_step = load_checkpoint(resume_path, student=student, teacher=teacher, optimizer=optimizer,
                                     ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, map_location=device)
        print(f"[mole] resumed from {resume_path} at step {start_step}")
    elif warm is not None:
        _apply_warmstart(student, teacher, warm)
        print(f"[mole] init-from: starting a fresh run at step 0 from {init_from}")

    # Provenance recorded in the manifest must reflect what ACTUALLY happened: the
    # source only when a fresh warm-start was applied, otherwise the value carried
    # by a real resume (preserved across resumes). Recording a merely-passed but
    # ignored --init-from is what previously let a stale dir masquerade as warm-started.
    if warm is not None:
        manifest_init = str(init_from)
    elif will_resume:
        manifest_init = _prior_init_from(run_dir)
    else:
        manifest_init = None
    _write_manifest(run_dir, cfg, dataset, mode, resume_path, start_step, init_from=manifest_init)

    # ---- Ctrl-C: checkpoint then exit cleanly ----
    state = {"stop": False}
    signal.signal(signal.SIGINT, lambda *_: state.update(stop=True))

    n_tokens = (d["model_size"] // m["patch_size"]) ** 2
    nct = m["num_class_tokens"]
    pred_shape = cfg["mask"]["pred_shape"]
    start_epoch, skip = divmod(start_step, steps_per_epoch)
    it = start_step
    total_steps = epochs * steps_per_epoch
    if start_step >= total_steps:
        print(f"[mole] run already complete: {start_step}/{total_steps} steps "
              f"({epochs} epochs). Nothing to do — raise train.epochs "
              f"(e.g. --set train.epochs={epochs * 2}) or use a fresh --output-dir.")
        return run_dir
    print(f"[mole] training {epochs} epochs × {steps_per_epoch} steps (start epoch {start_epoch}, step {it})")

    tb = _make_tb_writer(run_dir, bool(cfg["train"].get("tensorboard", True)))
    tb_every = max(1, int(cfg["train"].get("tb_every_steps", 10)))

    # Two persistent bars on fixed lines: outer = epochs (position 0), inner =
    # steps within the current epoch (position 1, reset() each epoch). Both created
    # ONCE — recreating the inner bar each epoch is what breaks nested rendering.
    # Equal-width labels + a shared bar_format so the two bars align vertically.
    label_w = max(len("epochs"), len(f"epoch {epochs}/{epochs}"))
    # Fixed {bar:72} width so the pipes align on both bars regardless of postfix
    # length (dynamic sizing would otherwise shrink the longer-postfix bar).
    bar_fmt = ("{desc} {percentage:3.0f}%|{bar:72}| {n_fmt}/{total_fmt} "
               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    epoch_bar = progress_bar(epochs, "epochs".rjust(label_w), unit="epoch", position=0,
                             initial=start_epoch, bar_format=bar_fmt)
    step_bar = progress_bar(steps_per_epoch, "steps".rjust(label_w), unit="step", position=1,
                            bar_format=bar_fmt)
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        loader = _build_loader(dataset, o["batch_size"], d["num_workers"], epoch, seed,
                               pin_memory=(device.type == "cuda"))
        step_bar.reset(total=steps_per_epoch)
        step_bar.set_description_str(f"epoch {epoch + 1}/{epochs}".rjust(label_w))
        if epoch == start_epoch and skip:
            step_bar.update(skip)  # already-done steps of a resumed epoch
        for i, (images, _) in enumerate(loader):
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
                step_bar.close()
                epoch_bar.close()
                if tb is not None:
                    tb.close()
                print(f"\n[mole] ERROR: loss became {loss.item()} at step {it} — training "
                      f"diverged; nothing saved.\n"
                      f"        likely causes:\n"
                      f"        • Apple MPS numerical instability — train on a CUDA GPU, not MPS\n"
                      f"        • learning rate too high — lower it (e.g. --set optim.lr=1e-4)\n"
                      f"        • resuming an already-diverged checkpoint — start fresh with a "
                      f"new --output-dir (delete the old run dir)")
                raise SystemExit(1)

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
            step_bar.update(1)
            step_bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr_sched[it - 1]:.2e}")

            if tb is not None and it % tb_every == 0:
                tb.add_scalar("loss/total", loss.item(), it)
                tb.add_scalar("loss/cls", float(all_loss["cls"]), it)
                tb.add_scalar("loss/patch", float(all_loss["patch"]), it)
                tb.add_scalar("sched/lr", lr_sched[it - 1], it)
                tb.add_scalar("sched/weight_decay", wd_sched[it - 1], it)
                tb.add_scalar("sched/momentum_teacher", mom_sched[it - 1], it)

            if it % int(cfg["train"]["save_every_steps"]) == 0:
                save_checkpoint(run_dir, student=student, teacher=teacher, optimizer=optimizer,
                                ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, global_step=it, config=cfg)
            if state["stop"]:
                save_checkpoint(run_dir, student=student, teacher=teacher, optimizer=optimizer,
                                ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, global_step=it, config=cfg)
                step_bar.close()
                epoch_bar.close()
                if tb is not None:
                    tb.close()
                print(f"\n[mole] interrupted — checkpointed at step {it} → {run_dir}/checkpoint.pth")
                return

        snap = epoch if (cfg["train"]["saveckp_epoch_freq"] and epoch
                         and epoch % cfg["train"]["saveckp_epoch_freq"] == 0) else None
        save_checkpoint(run_dir, student=student, teacher=teacher, optimizer=optimizer,
                        ibot_loss=ibot_loss, fp16_scaler=fp16_scaler, global_step=it, config=cfg,
                        epoch_snapshot=snap)
        epoch_bar.update(1)
        epoch_bar.set_postfix(loss=f"{float(loss.item()):.3f}")
        with (run_dir / "log.txt").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "step": it, "loss": float(loss.item())}) + "\n")

    step_bar.close()
    epoch_bar.close()
    if tb is not None:
        tb.close()
    print(f"[mole] done — {it} steps → {run_dir}/checkpoint.pth")
    return run_dir


def _write_manifest(run_dir, cfg, dataset, mode, resume_path, start_step, init_from=None):
    manifest = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "config_hash": config_hash(cfg),
        "parent_checkpoint": str(resume_path) if resume_path else None,
        "init_from": str(init_from) if init_from else None,
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
