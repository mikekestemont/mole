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
    output_dir: Path = typer.Argument(Path("prep"), help="Where cropped pages + QC are written."),
    method: str = typer.Option("yolo", help="Detector: 'yolo' (mole[detect]) or 'heuristic'."),
    padding: int = typer.Option(16, help="Padding (px) around the detected text zone."),
    conf: float = typer.Option(0.25, help="YOLO confidence threshold."),
    sample: Optional[int] = typer.Option(None, help="Process only a random N pages (quick QC)."),
    qc: Path = typer.Option(Path("outputs/prep_qc.html"), help="QC contact-sheet HTML path."),
) -> None:
    """Isolate the main handwritten text zone of each page and crop it (optional stage)."""
    from mole.prep import prep_folder

    try:
        records = prep_folder(input_dir, output_dir, method=method, padding=padding,
                              sample=sample, qc_html=qc, conf=conf)
    except ImportError as e:
        console.print(f"[red]Missing dependency for method '{method}': {e}[/red]")
        console.print("[yellow]For the YOLO detector: pip install 'mole[detect]'[/yellow]")
        raise typer.Exit(code=1)

    n_fb = sum(1 for r in records if r.fell_back)
    console.print(f"[green]✓ prepped {len(records)} pages → {output_dir}/images[/green]"
                  + (f" [yellow]({n_fb} fell back to whole page)[/yellow]" if n_fb else ""))
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
    seed: int = typer.Option(0, help="Random seed for reproducible grids."),
) -> None:
    """Preview augmentation strength as an image grid (CPU-only, seconds)."""
    from mole.data.augment import augview as _augview

    presets = [preset] if preset is not None else None
    out = _augview(str(folder), str(output), n_images=n_images, n_views=n_views,
                   presets=presets, seed=seed, window_size=window_size)
    console.print(f"[green]✓ wrote augmentation grid → {out}[/green]")


# -------------------------------------------------------------------------- train
@app.command()
def train(
    config: Path = typer.Argument(..., help="YAML config file."),
    output_dir: Optional[Path] = typer.Option(None, help="Run directory."),
    mode: str = typer.Option("scratch", help="'scratch' or 'continual' (replay)."),
    resume: Optional[Path] = typer.Option(None, help="Resume from a run directory."),
    set_: list[str] = typer.Option([], "--set", help="Override a config leaf, e.g. optim.lr=1e-4."),
) -> None:
    """Pretrain (or continually update) the base model with AttMask."""
    _todo("train", phase=4)


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
    pooling: Pooling = typer.Option(Pooling.MEAN, help="Pooling strategy."),
    whiten: bool = typer.Option(False, help="Apply PCA-whitening."),
) -> None:
    """Extract page embeddings (mean/cls/vlad/patches) with lineage stamping."""
    _todo("embed", phase=5)


# --------------------------------------------------------------------------- eval
@app.command()
def eval(  # noqa: A001 - deliberately mirrors the subcommand name
    embeddings: Path = typer.Argument(..., help="Embeddings file to evaluate."),
    datasets_root: Path = typer.Argument(..., help="Datasets root (for labels.csv)."),
) -> None:
    """Run the retrieval benchmark from partial labels (mAP, top-k, cross-dataset)."""
    _todo("eval", phase=6)


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
