"""The `ftir-map` command line: errors are one line on stderr and exit status 1, never a traceback."""

import subprocess
import sys

import pytest

from ftir_map.cli import main

from .omnic_writer import write_map
from .test_omnic import GRID, spectra


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "ftir_map.cli", *args], capture_output=True, text=True, check=False)


@pytest.fixture
def map_bytes(tmp_path):
    write_map(tmp_path / "g.map", spectra(6), GRID, kind="area")
    return (tmp_path / "g.map").read_bytes()


@pytest.mark.parametrize("cut", [0x20, 0x130, 0x140, 0x200, 0x300, -40])
def test_info_on_a_truncated_map_is_one_line_and_status_1(tmp_path, map_bytes, cut):
    path = tmp_path / "cut.map"
    path.write_bytes(map_bytes[:cut])
    result = run_cli("info", str(path))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert result.stderr.startswith("ftir-map: ") and len(result.stderr.splitlines()) == 1


@pytest.mark.parametrize("cut", [0x20, 0x130, 0x300])
def test_export_and_unpack_on_a_truncated_map_do_not_traceback(tmp_path, map_bytes, cut):
    path = tmp_path / "cut.map"
    path.write_bytes(map_bytes[:cut])
    assert main(["export", str(path), "--out", str(tmp_path / "out")]) == 1
    result = run_cli("unpack", str(path), "--out", str(tmp_path / "project"))
    assert result.returncode == 0 and "skipped" in result.stdout and "Traceback" not in result.stderr


def test_export_into_a_folder_that_is_not_empty_names_the_command_line_option(tmp_path, map_bytes, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.txt").write_text("x")
    assert main(["export", str(tmp_path / "g.map"), "--out", str(out)]) == 1
    error = capsys.readouterr().err
    assert "--overwrite" in error and "overwrite=True" not in error
