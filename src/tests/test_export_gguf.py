"""Converter-interpreter resolution for the GGUF export.

llama.cpp's convert_hf_to_gguf.py needs transformers==5.5.1, which pins
tokenizers<=0.23.0 — a version nanobeard cannot install. So the converter runs
under a *different* interpreter, and picking that interpreter is the thing worth
testing: getting it wrong silently falls back to an env that cannot import gguf,
and the failure surfaces deep inside someone else's script.

hf/ holds standalone scripts, not an importable package, so load the module by
path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "hf" / "export_gguf.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_gguf_under_test", _SRC)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def eg():
    return _load()


def test_explicit_flag_wins_over_everything(eg, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOBEARD_CONVERTER_PYTHON", "/from/env/python")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    assert eg.resolve_converter_python("/explicit/python", tmp_path) == "/explicit/python"


def test_env_var_beats_the_checkout_venv(eg, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOBEARD_CONVERTER_PYTHON", "/from/env/python")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    assert eg.resolve_converter_python(None, tmp_path) == "/from/env/python"


@pytest.mark.parametrize("venv", [".venv", "venv"])
def test_falls_back_to_a_venv_inside_the_llama_cpp_checkout(eg, tmp_path, monkeypatch, venv):
    monkeypatch.delenv("NANOBEARD_CONVERTER_PYTHON", raising=False)
    bin_dir = tmp_path / venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").touch()
    assert eg.resolve_converter_python(None, tmp_path) == str(bin_dir / "python")


def test_last_resort_is_this_interpreter(eg, tmp_path, monkeypatch):
    monkeypatch.delenv("NANOBEARD_CONVERTER_PYTHON", raising=False)
    assert eg.resolve_converter_python(None, tmp_path) == sys.executable


def test_dep_check_passes_for_an_env_that_has_them(eg, monkeypatch):
    # Probe a dependency set this interpreter definitely satisfies.
    monkeypatch.setattr(eg, "CONVERTER_DEPS", ("json", "os"))
    eg.check_converter_deps(sys.executable)  # must not raise


def test_dep_check_exits_with_the_install_command(eg, monkeypatch):
    monkeypatch.setattr(eg, "CONVERTER_DEPS", ("definitely_not_a_real_module_xyz",))
    with pytest.raises(SystemExit) as exc:
        eg.check_converter_deps(sys.executable)
    msg = str(exc.value)
    assert "transformers==5.5.1" in msg
    # Pointing at nanobeard's own env is the mistake worth naming explicitly.
    assert "tokenizers<=0.23.0" in msg


def test_dep_check_names_the_foreign_env_it_probed(eg, tmp_path, monkeypatch):
    monkeypatch.setattr(eg, "CONVERTER_DEPS", ("definitely_not_a_real_module_xyz",))
    other = tmp_path / "python"
    other.symlink_to(sys.executable)
    with pytest.raises(SystemExit) as exc:
        eg.check_converter_deps(str(other))
    assert str(other) in str(exc.value)
    assert os.fspath(other) in str(exc.value)
