"""MOLE command-line interface.

Every command maps to a documented function in the Python API (feature parity).
During the phased build, commands whose backing phase has not landed yet print a
clear "not implemented" notice and exit cleanly, so ``mole --help`` and each
subcommand's ``--help`` render fully from day one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from mole import __version__
from mole.data.augment import AugPreset
from mole.embed.pooling import Pooling

app = typer.Typer(
    name="mole",
    help="Continual self-supervised handwriting embeddings for premodern documents.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Inspect the model lineage registry.", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()


def _todo(command: str, phase: int) -> None:
    """Report that a command's implementation phase has not landed yet."""
    console.print(
        f"[yellow]⏳ `mole {command}` is scaffolded but not implemented yet "
        f"(lands in Phase {phase}).[/yellow]"
    )
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mole {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the mole version and exit.",
    ),
) -> None:
    """MOLE — continual self-supervised embeddings for premodern handwriting."""


# --------------------------------------------------------------------------- prep
@app.command()
def prep(
    input_dir: Path = typer.Argument(..., help="Folder of page images to preprocess."),
    zones_out: Optional[Path] = typer.Option(None, help="zones.json path (default: <input_dir>/zones.json)."),
    method: str = typer.Option("yolo", help="Detector: 'yolo' (mole[detect]) or 'heuristic'."),
    padding: int = typer.Option(16, help="Padding (px) around the detected text zone."),
    conf: float = typer.Option(0.25, help="YOLO confidence threshold."),
    sample: Optional[int] = typer.Option(None, help="Process only a random N pages (quick QC)."),
    qc: Path = typer.Option(Path("outputs/prep_qc.html"), help="QC contact-sheet HTML path."),
    write_crops: Optional[Path] = typer.Option(None, help="Also materialise cropped images into this folder (opt-in)."),
    from_zones: bool = typer.Option(False, "--from-zones", help="Rebuild QC (+crops) from existing zones.json — no detector, no GPU."),
    binarize: str = typer.Option("none", help="Binarize images first: 'none' or 'sauvola' (adaptive; for camera photos)."),
    binarize_out: Optional[Path] = typer.Option(None, help="Output dir for binarized images (default: <input_dir>-bin)."),
    sauvola_window: int = typer.Option(25, help="Sauvola local window in px (odd)."),
    sauvola_k: float = typer.Option(0.2, help="Sauvola k — higher = more aggressive/thinner ink."),
    max_side: int = typer.Option(0, help="Downscale longest side to <= N px before binarizing (0 = off, never upsamples)."),
) -> None:
    """Detect the main handwritten text zone of each page and store coordinates.

    Writes a zones.json manifest (reused by augview/train/embed) into the dataset
    folder; the detector runs once. Cropped images are opt-in via --write-crops.
    Use --from-zones to just re-view results from a stored manifest (fast, no GPU).

    With --binarize sauvola, instead binarizes the images (black-on-white) into a
    new folder + a QC sheet; combine with --sample N to preview before committing.
    """
    if binarize != "none":
        from mole.prep.binarize import binarize_folder

        out_dir = binarize_out or input_dir.parent / f"{input_dir.name}-bin"
        recs = binarize_folder(input_dir, out_dir, method=binarize, window=sauvola_window,
                               k=sauvola_k, max_side=max_side or None, sample=sample, qc_html=qc)
        if sample is not None:
            console.print(f"[yellow]preview only ({len(recs)} images) — nothing written; "
                          f"tune --sauvola-window/--sauvola-k, then re-run without --sample[/yellow]")
        else:
            console.print(f"[green]✓ binarized {len(recs)} images → {out_dir}[/green]")
        console.print(f"[green]✓ QC sheet → {qc}[/green]")
        return

    from mole.prep import prep_folder, qc_from_zones

    if from_zones:
        records = qc_from_zones(input_dir, zones_out=zones_out, qc_html=qc, write_crops=write_crops)
        console.print(f"[green]✓ rebuilt QC for {len(records)} pages from zones.json → {qc}[/green]")
        if write_crops:
            console.print(f"[green]✓ cropped images → {write_crops}[/green]")
        return

    try:
        manifest, records = prep_folder(input_dir, zones_out=zones_out, method=method,
                                        padding=padding, sample=sample, qc_html=qc,
                                        conf=conf, write_crops=write_crops)
    except ImportError as e:
        console.print(f"[red]Missing dependency for method '{method}': {e}[/red]")
        console.print("[yellow]For the YOLO detector: pip install 'mole[detect]'[/yellow]")
        raise typer.Exit(code=1)

    zpath = zones_out or (input_dir / "zones.json")
    n_fb = sum(1 for r in records if r.fell_back)
    console.print(f"[green]✓ {len(records)} pages → zones stored in {zpath}[/green]"
                  + (f" [yellow]({n_fb} fell back to whole page)[/yellow]" if n_fb else ""))
    if write_crops:
        console.print(f"[green]✓ cropped images → {write_crops}[/green]")
    console.print(f"[green]✓ QC sheet → {qc}[/green]")


# ------------------------------------------------------------------------ augview
@app.command()
def augview(
    folder: Path = typer.Argument(..., help="Folder of images to sample from."),
    output: Path = typer.Option(Path("outputs/augview.html"), help="Output HTML grid."),
    n_images: int = typer.Option(5, help="Number of source images."),
    n_views: int = typer.Option(5, help="Augmented views per image."),
    preset: Optional[AugPreset] = typer.Option(
        None, help="Preview one preset only (default: all three side by side)."),
    window_size: int = typer.Option(512, help="Patch-window size sampled before augmenting."),
    zones: Optional[Path] = typer.Option(None, help="zones.json to restrict sampling (default: auto-discover in folder)."),
    no_zones: bool = typer.Option(False, "--no-zones", help="Ignore any zones.json; sample the whole page."),
    seed: int = typer.Option(0, help="Random seed for reproducible grids."),
) -> None:
    """Preview augmentation strength as an image grid (CPU-only, seconds).

    Restricts sampling to the prep text zone when a zones.json is found in the
    folder (or given via --zones), so windows never come from background/clutter.
    """
    from mole.data.augment import augview as _augview

    presets = [preset] if preset is not None else None
    out = _augview(str(folder), str(output), n_images=n_images, n_views=n_views,
                   presets=presets, seed=seed, window_size=window_size,
                   zones_path=str(zones) if zones else None, use_zones=not no_zones)
    console.print(f"[green]✓ wrote augmentation grid → {out}[/green]")


# -------------------------------------------------------------------------- train
@app.command()
def train(
    config: Path = typer.Argument(..., help="YAML config file."),
    output_dir: Optional[Path] = typer.Option(None, help="Run directory (overrides config)."),
    mode: str = typer.Option("scratch", help="'scratch' or 'continual' (replay: Phase 7)."),
    resume: Optional[Path] = typer.Option(None, help="Resume from a checkpoint (auto if run dir has one)."),
    init_from: Optional[Path] = typer.Option(
        None, "--init-from",
        help="Warm-start weights from a foreign (original AttMask/iBOT) or mole checkpoint "
             "(fresh run at step 0; ignored when resuming)."),
    set_: list[str] = typer.Option([], "--set", help="Override a config leaf, e.g. optim.lr=1e-4."),
) -> None:
    """Pretrain (or continually update) the base model with AttMask."""
    from mole.selfsup.train import train as _train

    _train(config, output_dir=output_dir, mode=mode, resume=resume, overrides=list(set_),
           init_from=init_from)


# ----------------------------------------------------------------------- finetune
@app.command()
def finetune(
    config: Path = typer.Argument(..., help="YAML config file."),
    base_checkpoint: Path = typer.Argument(..., help="Base checkpoint to branch from."),
    output_dir: Optional[Path] = typer.Option(None, help="Run directory (a new branch)."),
    set_: list[str] = typer.Option([], "--set", help="Override a config leaf."),
) -> None:
    """Branch a dataset-specific finetune from a base checkpoint (never mutates base)."""
    _todo("finetune", phase=7)


# -------------------------------------------------------------------------- embed
@app.command()
def embed(
    checkpoint: Path = typer.Argument(..., help="Model checkpoint to extract with."),
    input_dir: Path = typer.Argument(..., help="Folder of images to embed."),
    output: Path = typer.Argument(..., help="Output .npy/.parquet path."),
    pooling: Pooling = typer.Option(Pooling.VLAD, help="Pooling strategy (default: vlad)."),
    whiten: bool = typer.Option(False, help="Apply PCA-whitening (fixed-vector poolings)."),
    whiten_dim: Optional[int] = typer.Option(
        None, "--whiten-dim",
        help="PCA-whiten AND reduce to this many dims (implies --whiten; e.g. 384 for the "
             "writer-retrieval VLAD->384 recipe)."),
    batch_size: int = typer.Option(32, help="Windows per forward pass."),
    vlad_clusters: int = typer.Option(64, help="VLAD codebook size (pooling=vlad)."),
    seed: int = typer.Option(0, help="VLAD k-means seed (reproducible codebook)."),
    device: Optional[str] = typer.Option(None, help="Force device (cuda/mps/cpu); default auto."),
    foreground: bool = typer.Option(
        False, "--foreground/--no-foreground",
        help="Drop background patch tokens before patches/vlad pooling."),
    foreground_method: str = typer.Option(
        "intensity", help="Foreground test: 'intensity' (Raven, white-bg/binarized) or "
                          "'contrast' (local std — works on parchment/colour)."),
    foreground_threshold: Optional[float] = typer.Option(
        None, help="Keep threshold; default 0.02 for intensity (mean<1-thr), 0.05 for "
                   "contrast (std>thr)."),
    vlad_intra_norm: bool = typer.Option(
        True, "--vlad-intra-norm/--no-vlad-intra-norm",
        help="Per-cluster intra-normalisation in VLAD; use --no-vlad-intra-norm for Raven-parity."),
    invert: Optional[bool] = typer.Option(
        None, "--invert/--no-invert",
        help="Negate intensity at load (white-on-black -> black-on-white). Default: inherit "
             "the training value from the checkpoint."),
    codebook_from: Optional[Path] = typer.Option(
        None, "--codebook-from",
        help="Reuse a saved .codebook.npy (pooling=vlad) instead of fitting one on this set — "
             "e.g. a codebook fit on the training split (fit-on-train / apply-on-test)."),
    set_: list[str] = typer.Option([], "--set", help="Override embed geometry, e.g. window_size=384."),
) -> None:
    """Extract page embeddings (mean/cls/vlad/patches) with lineage stamping.

    Raven-parity VLAD baseline: --pooling vlad --vlad-clusters 100 --foreground
    --no-vlad-intra-norm.
    """
    from mole.embed import embed as _embed

    _embed(checkpoint, input_dir, output, pooling=pooling, whiten=whiten,
           overrides=list(set_), batch_size=batch_size, vlad_clusters=vlad_clusters,
           seed=seed, device=device, foreground=foreground,
           foreground_threshold=foreground_threshold, foreground_method=foreground_method,
           vlad_intra_norm=vlad_intra_norm,
           invert=invert, codebook_from=codebook_from, whiten_dim=whiten_dim)


# ---------------------------------------------------------------------------- viz
@app.command()
def viz(
    embeddings: Path = typer.Argument(..., help="Embeddings .npy (its .mapping.json sidecar is read too)."),
    out: Optional[Path] = typer.Option(None, help="Output HTML (default: <embeddings>.viz.html)."),
    method: str = typer.Option("auto", help="2D projection: auto (PCA→UMAP) | pca | tsne | umap."),
    pca_dim: int = typer.Option(150, help="PCA pre-reduction dims before UMAP/t-SNE."),
    color: str = typer.Option("dataset", help="Colour points by: dataset|hand|none."),
    color_regex: Optional[str] = typer.Option(None, help=r"Colour by a filename capture group, e.g. '_(\d{4})-' for year."),
    seed: int = typer.Option(0, help="Projection seed (reproducible)."),
) -> None:
    """Project an embeddings file to 2D and write an interactive HTML scatter."""
    from mole.viz import plot_embeddings

    out_path, used = plot_embeddings(embeddings, out=out, method=method, color=color,
                                     color_regex=color_regex, seed=seed, pca_dim=pca_dim)
    console.print(f"[green]✓ {used} scatter → {out_path}[/green]")


# --------------------------------------------------------------------------- eval
@app.command()
def eval(  # noqa: A001 - deliberately mirrors the subcommand name
    embeddings: Path = typer.Argument(..., help="Embeddings .npy to evaluate (page-level)."),
    datasets_root: Path = typer.Argument(..., help="Dataset dir or root holding labels.csv."),
    metric: str = typer.Option("cosine", help="Ranking metric: cosine | euclidean."),
    topk: str = typer.Option("1,5,10", help="Comma-separated Top-k cutoffs to report."),
    out: Optional[Path] = typer.Option(None, help="JSON report path (default: <embeddings>.eval.json)."),
) -> None:
    """Retrieval benchmark from partial labels: mAP, Top-k, cross-dataset breakdown."""
    from mole.eval import evaluate, format_report

    ks = tuple(int(k) for k in topk.split(",") if k.strip())
    result = evaluate(embeddings, datasets_root, metric=metric, topk=ks, out=out)
    console.print(format_report(result))


# ------------------------------------------------------------------- models list/show
@models_app.command("list")
def models_list(
    models_root: Path = typer.Argument(Path("models"), help="Models root directory."),
) -> None:
    """Print the model lineage as a tree."""
    _todo("models list", phase=6)


@models_app.command("show")
def models_show(
    model_id: str = typer.Argument(..., help="Model ID, e.g. base@v2 or base@v3/stgallen@v1."),
    models_root: Path = typer.Argument(Path("models"), help="Models root directory."),
) -> None:
    """Print full provenance of one checkpoint."""
    _todo("models show", phase=6)


if __name__ == "__main__":
    app()
