"""The `ftir-map` command line: errors are one line on stderr and exit status 1, never a traceback."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ftir_map import read_omnic
from ftir_map.cli import build_parser, format_catalog, format_unpack_line, main

from .omnic_writer import write_map
from .test_omnic import DATA, GRID, spectra

SRC = str(Path(__file__).resolve().parents[1] / "src")
CLI_DOC = (Path(__file__).resolve().parents[1] / "docs/site/cli.md").read_text(encoding="utf-8")
POINT_MAP = "GCB_1C_GC52_CPX best points-50x50um_256avg.map"
GRID_MAP = "GCB_1C_ol54_ultrafast map_75um aperture and step.map"
LINE_MAP = "GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map"


def run_cli(*args):
    env = {**os.environ, "PYTHONPATH": SRC}  # the copy under test, not an installed one
    return subprocess.run(
        [sys.executable, "-m", "ftir_map.cli", *args], capture_output=True, text=True, check=False, env=env
    )


def real(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not in tests/data")
    return path


def doc_block(heading_line):
    """The text of the first ```console block that follows `heading_line` in docs/site/cli.md, without its `$` line."""
    after = CLI_DOC.split(heading_line, 1)[1]
    block = after.split("```console\n", 1)[1].split("```", 1)[0]
    return block.split("\n", 1)[1]


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


# ------------------------------------------------------------------ success paths, run in-process so mutants show
def run_main(capsys, *args):
    status = main([str(a) for a in args])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def info_fields(text):
    return dict(line.split(": ", 1) for line in text.splitlines() if not line.startswith("  ") and ": " in line)


def csv_axis_ends(folder):
    """First and last wavenumber of OMNIC's own CSV export of the point map (7 significant digits)."""
    rows = sorted(folder.glob("*avg000?.CSV"))  # the six spectra, not the background (avgbkg1)
    first = [line.split(",")[0] for line in rows[0].read_text().splitlines() if line.strip()]
    return float(first[0]), float(first[-1]), len(first), len(rows)


def test_info_on_the_point_map_matches_omnics_csv_export(capsys):
    status, out, err = run_main(capsys, "info", real(POINT_MAP))
    fields = info_fields(out)
    low, high, n_values, n_spectra = csv_axis_ends(DATA / "cpx52_best_points_csv")
    match = re.fullmatch(r"([\d.]+) to ([\d.]+) cm-1, (\d+) values", fields["axis"])
    assert match is not None
    assert (status, err) == (0, "")
    assert (fields["file"], fields["map kind"], fields["points"], fields["data kind"]) == (
        POINT_MAP,
        "points",
        str(n_spectra),
        "absorbance",
    )
    assert n_spectra == 6 and int(match[3]) == n_values == 1725
    # the CLI prints 6 significant digits, the CSV has 7: relative tolerance 1e-5
    assert float(match[1]) == pytest.approx(low, rel=1e-5) and float(match[2]) == pytest.approx(high, rel=1e-5)
    assert fields["images"] == "2"


def test_info_on_the_grid_map_prints_the_sample_in_cli_md(capsys):
    status, out, err = run_main(capsys, "info", real(GRID_MAP))
    assert (status, err) == (0, "")
    assert out == doc_block("For example:")  # docs/site/cli.md shows this output; the doc is the oracle
    assert info_fields(out)["map kind"] == "grid (13 rows x 11 columns)" and info_fields(out)["points"] == "143"


def test_info_on_the_command_line_of_a_subprocess_matches_in_process(capsys):
    status, out, _ = run_main(capsys, "info", real(POINT_MAP))
    result = run_cli("info", str(real(POINT_MAP)))
    assert (status, result.returncode, result.stderr) == (0, 0, "")
    assert result.stdout == out


def test_export_writes_the_documented_files(tmp_path, capsys):
    out = tmp_path / "exported"
    status, text, err = run_main(capsys, "export", real(POINT_MAP), "--out", out)
    assert (status, err, text) == (0, "", f"wrote {out}\n")
    names = {p.name for p in out.iterdir()}
    assert names == {"spectra.zarr", "points.csv", "metadata.json", "images", "README.txt"}  # cli.md defaults
    assert len(list((out / "images").glob("*.png"))) == 2  # the point map has 2 images
    assert len((out / "points.csv").read_text().splitlines()) == 1 + 6  # header and 6 spectra
    assert json.loads((out / "metadata.json").read_text())["n_points"] == 6


def test_export_options_choose_the_formats(tmp_path, capsys):
    out = tmp_path / "exported"
    args = ("export", real(POINT_MAP), "--out", out, "--spectra", "npz", "--metadata", "json", "--points", "json")
    assert run_main(capsys, *args, "--images", "none")[0] == 0
    assert {p.name for p in out.iterdir()} == {"spectra.npz", "points.json", "metadata.json", "README.txt"}
    with np.load(out / "spectra.npz") as archive:
        assert archive["spectra"].shape == (6, 1725)  # 6 spectra with 1725 axis values
    assert run_main(capsys, *args, "--images", "tiff", "--overwrite")[0] == 0
    assert len(list((out / "images").glob("*.tif*"))) == 2


@pytest.fixture
def three_real_maps(tmp_path):
    folder = tmp_path / "library"
    folder.mkdir()
    for name in (POINT_MAP, GRID_MAP, LINE_MAP):
        (folder / name).write_bytes(real(name).read_bytes())
    return folder


# (file name -> map folder name by the rule in unpacked_format.md, spectra, kind, images) from the files' own info output
UNPACKED = {
    POINT_MAP: ("GCB_1C_GC52_CPX_best_points-50x50um_256avg", 6, "points", 2),
    GRID_MAP: ("GCB_1C_ol54_ultrafast_map_75um_aperture_and_step", 143, "grid", 6),
    LINE_MAP: ("GCB_4_Region_2_grain2_profile_2_50x100um_mosaic_image_test", 21, "line", 6),
}


def test_unpack_prints_a_line_per_map_and_the_summary(three_real_maps, tmp_path, capsys):
    out = tmp_path / "project"
    status, text, err = run_main(capsys, "unpack", three_real_maps, "--out", out)
    expected = {f"unpacked  {name}: {n} points, {kind}, {k} images" for name, n, kind, k in UNPACKED.values()}
    assert (status, err) == (0, "")
    lines = text.splitlines()
    assert set(lines[:-1]) == expected and len(lines) == 4
    assert lines[-1] == f"3 unpacked, 0 unchanged, 0 skipped -> {out}"
    status, text, err = run_main(capsys, "unpack", three_real_maps, "--out", out)
    lines = text.splitlines()
    assert {line.replace("unchanged", "unpacked ", 1) for line in lines[:-1]} == expected
    assert lines[-1] == f"0 unpacked, 3 unchanged, 0 skipped -> {out}"
    assert (status, err) == (0, "")


def test_unpack_options_reach_the_unpacker(three_real_maps, tmp_path, capsys):
    out = tmp_path / "project"
    point = three_real_maps / POINT_MAP
    assert run_main(capsys, "unpack", point, "--out", out, "--spectra", "csv", "--png", "--copy-source")[0] == 0
    folder = out / "maps" / UNPACKED[POINT_MAP][0]
    assert (folder / "spectra.csv").is_file() and (folder / "source" / POINT_MAP).is_file()
    assert len(list((folder / "images" / "png").glob("*.png"))) == 2
    status, text, _ = run_main(capsys, "unpack", point, "--out", out, "--overwrite")
    assert status == 0 and text.splitlines()[-1] == f"1 unpacked, 0 unchanged, 0 skipped -> {out}"
    assert (folder / "spectra.zarr").is_dir() and not (folder / "spectra.csv").exists()


def test_catalog_lists_every_map(three_real_maps, tmp_path, capsys):
    out = tmp_path / "project"
    run_main(capsys, "unpack", three_real_maps, "--out", out)
    status, text, err = run_main(capsys, "catalog", out)
    header, *rows = (line.split() for line in text.splitlines())
    assert (status, err) == (0, "")
    assert header == ["name", "map_kind", "n_points", "axis", "data_kind", "n_images", "status"]
    got = {row[0]: row for row in rows}
    assert len(rows) == 3
    # name, kind, points, "<low>-<high>", unit, data kind, images, status (axis ends as printed with 6 digits)
    for name, n, kind, k in UNPACKED.values():
        _, row_kind, row_n, axis, unit, data_kind, row_images, row_status = got[name]
        low, high = (float(v) for v in axis.split("-"))
        assert (row_kind, row_n, unit, data_kind, row_images, row_status) == (
            kind,
            str(n),
            "cm-1",
            "absorbance",
            str(k),
            "ok",
        )
        assert 600 < low < 700 and 3999 < high < 4004  # cli.md: the maps start near 675-679 and end near 4000 cm-1
    assert got[UNPACKED[GRID_MAP][0]][3] == "678.849-4003.67"  # the axis in the cli.md sample for this map


def test_format_unpack_line_pads_the_action_to_nine_characters():
    row = {"name": "m", "n_points": 6, "map_kind": "grid", "n_images": 1, "status": "ok"}
    assert format_unpack_line(row, "unpacked") == "unpacked  m: 6 points, grid, 1 images"
    assert format_unpack_line(row, "unchanged") == "unchanged m: 6 points, grid, 1 images"
    skipped = {"name": "b", "status": "skipped: no spectra"}
    assert format_unpack_line(skipped, "skipped") == "skipped   b: no spectra"


def test_format_catalog_aligns_columns_and_leaves_missing_cells_empty():
    ok = {
        "name": "alpha",
        "map_kind": "grid",
        "n_points": 6,
        "axis_min": 650.0,
        "axis_max": 4000.0,
        "axis_unit": "cm-1",
        "data_kind": "absorbance",
        "n_images": 0,
        "status": "ok",
    }
    skipped = dict.fromkeys(ok) | {"name": "b", "status": "skipped: x"}
    lines = format_catalog(SimpleNamespace(rows=[ok, skipped])).splitlines()  # ty: ignore[invalid-argument-type]
    assert lines == [
        "name   map_kind  n_points  axis           data_kind   n_images  status",
        "alpha  grid      6         650-4000 cm-1  absorbance  0         ok",
        "b".ljust(64) + "skipped: x",  # name, 8, 8, 13, 10 and 8 wide columns, each followed by two spaces
    ]


# ------------------------------------------------------------------ the parser against the table in cli.md
def choices_in(text, option):
    """The `{a,b,c}` choices that follow `option` in a usage or help text."""
    match = re.search(rf"{re.escape(option)} \{{([^}}]*)\}}", text)
    assert match is not None
    return match[1].split(",")


@pytest.mark.parametrize(
    ("command", "option"),
    [
        ("export", "--spectra"),
        ("export", "--metadata"),
        ("export", "--points"),
        ("export", "--images"),
        ("unpack", "--spectra"),
    ],
)
def test_option_choices_equal_the_ones_in_cli_md(command, option, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "300")  # no line wrapping in the help text
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, "--help"])
    usage = CLI_DOC.split(f"$ ftir-map {command} --help", 1)[1].split("```", 1)[0]
    assert choices_in(capsys.readouterr().out, option) == choices_in(usage, option)


def test_option_defaults_equal_the_ones_in_cli_md():
    export = build_parser().parse_args(["export", "m.map", "--out", "o"])
    assert (export.spectra, export.metadata, export.points, export.images, export.overwrite) == (
        "zarr",
        "json",
        "csv",
        "png",
        False,
    )  # "The defaults are --spectra zarr, --metadata json, --points csv and --images png."
    unpack = build_parser().parse_args(["unpack", "a.map", "b", "--out", "p"])
    assert (unpack.sources, unpack.out, unpack.spectra) == (["a.map", "b"], "p", "zarr")
    assert (unpack.png, unpack.copy_source, unpack.overwrite) == (False, False, False)  # flags, off unless given
    flags = build_parser().parse_args(["unpack", "a.map", "--out", "p", "--png", "--copy-source", "--overwrite"])
    assert (flags.png, flags.copy_source, flags.overwrite) == (True, True, True)
    assert build_parser().parse_args(["catalog", "p"]).project == "p"
    assert build_parser().parse_args(["info", "f.map"]).file == "f.map"


def test_parser_refuses_a_missing_command_and_a_missing_out(capsys):
    for argv in (
        [],
        ["export", "m.map"],
        ["unpack", "a.map"],
        ["unpack", "--out", "p"],
        ["export", "m", "--out", "o", "--spectra", "tsv"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(argv)
        assert exit_info.value.code == 2
    capsys.readouterr()


# ------------------------------------------------------------------ a damaged catalog is one line, status 1
@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "lib").mkdir()
    write_map(tmp_path / "lib" / "g.map", spectra(6), GRID, kind="area")
    out = tmp_path / "project"
    assert main(["unpack", str(tmp_path / "lib"), "--out", str(out)]) == 0
    return out


def damage_missing_maps(content):
    del content["maps"]


def damage_missing_axis_min(content):
    del content["maps"][0]["axis_min"]


def damage_missing_sha256(content):
    del content["maps"][0]["sha256"]


def damage_rows_not_objects(content):
    content["maps"] = ["g"]


@pytest.mark.parametrize(
    "damage", [damage_missing_maps, damage_missing_axis_min, damage_missing_sha256, damage_rows_not_objects]
)
@pytest.mark.parametrize("command", ["catalog", "unpack"])
def test_damaged_catalog_json_is_one_line_and_status_1(project_dir, tmp_path, capsys, damage, command):
    path = project_dir / "catalog.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    damage(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    args = ["catalog", project_dir] if command == "catalog" else ["unpack", tmp_path / "lib", "--out", project_dir]
    status, out, err = run_main(capsys, *args)
    assert status == 1 and out == ""
    assert "Traceback" not in err and err.startswith("ftir-map: ") and len(err.splitlines()) == 1
    assert "catalog.json" in err


@pytest.mark.parametrize("command", ["catalog", "unpack"])
def test_catalog_json_that_is_not_json_is_one_line_and_status_1(project_dir, tmp_path, capsys, command):
    (project_dir / "catalog.json").write_text("{not json", encoding="utf-8")
    args = ["catalog", project_dir] if command == "catalog" else ["unpack", tmp_path / "lib", "--out", project_dir]
    status, _, err = run_main(capsys, *args)
    assert status == 1 and err.startswith("ftir-map: ") and len(err.splitlines()) == 1 and "Traceback" not in err
    assert "not valid JSON" in err


def test_a_damaged_catalog_is_one_line_through_the_real_command(project_dir):
    content = json.loads((project_dir / "catalog.json").read_text(encoding="utf-8"))
    del content["maps"]
    (project_dir / "catalog.json").write_text(json.dumps(content), encoding="utf-8")
    result = run_cli("catalog", str(project_dir))
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert result.stderr.startswith("ftir-map: ") and len(result.stderr.splitlines()) == 1


def test_read_omnic_and_cli_agree_on_the_number_of_spectra(tmp_path, capsys):
    path = real(POINT_MAP)
    assert info_fields(run_main(capsys, "info", path)[1])["points"] == str(read_omnic(path).n_points)
