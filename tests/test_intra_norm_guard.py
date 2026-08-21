"""Intra-normalization stays opt-in (default OFF; it hurts balanced collections), and
the index-mixing guard catches plain/intra VLAD sitting in one directory."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mole.embed.extract import _warn_on_version_mismatch, embed
from mole.embed.pooling import Pooling


def test_cli_default_is_plain_vlad():
    # Decision (2026-08-21): intra-norm is opt-in, not the default — it trips the §4.2
    # guardrail on balanced collections (Leroy) in the frozen-codebook config.
    from mole.cli.main import embed as cli_embed
    default = inspect.signature(cli_embed).parameters["vlad_intra_norm"].default
    assert default.default is False               # typer.Option(False, ...)


def test_embed_api_default_is_plain_vlad():
    default = inspect.signature(embed).parameters["vlad_intra_norm"].default
    assert default is False


def test_guard_warns_on_mixed_intra_norm(tmp_path, capsys):
    # A plain-VLAD sidecar in the dir must warn when the current run is intra-normalised
    # (the silent mix the opt-in flag could otherwise create).
    (tmp_path / "old.mapping.json").write_text(json.dumps(
        {"model_id": "m@abcd1234+step0", "pooling": "vlad", "vlad_intra_norm": False}))
    _warn_on_version_mismatch(tmp_path, "m@abcd1234+step0",
                              pooling=Pooling.VLAD, vlad_intra_norm=True)
    msg = capsys.readouterr().out
    assert "vlad_intra_norm" in msg and "different spaces" in msg


def test_guard_silent_when_intra_norm_matches(tmp_path, capsys):
    (tmp_path / "old.mapping.json").write_text(json.dumps(
        {"model_id": "m@abcd1234+step0", "pooling": "vlad", "vlad_intra_norm": True}))
    _warn_on_version_mismatch(tmp_path, "m@abcd1234+step0",
                              pooling=Pooling.VLAD, vlad_intra_norm=True)
    assert capsys.readouterr().out == ""


def test_guard_ignores_norm_across_poolings(tmp_path, capsys):
    # vlad_intra_norm only matters VLAD-vs-VLAD; a mean sidecar must not trip it.
    (tmp_path / "old.mapping.json").write_text(json.dumps(
        {"model_id": "m@abcd1234+step0", "pooling": "mean"}))
    _warn_on_version_mismatch(tmp_path, "m@abcd1234+step0",
                              pooling=Pooling.VLAD, vlad_intra_norm=True)
    assert capsys.readouterr().out == ""


def test_guard_still_warns_on_model_mismatch(tmp_path, capsys):
    # The original model-version guard is unchanged.
    (tmp_path / "old.mapping.json").write_text(json.dumps(
        {"model_id": "other@deadbeef+step0", "pooling": "mean"}))
    _warn_on_version_mismatch(tmp_path, "m@abcd1234+step0")
    assert "DIFFERENT model" in capsys.readouterr().out
