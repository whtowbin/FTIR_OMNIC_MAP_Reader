"""Unpacked maps and project folders: round trip, open formats, idempotent re-runs, the command line."""

import csv
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import ftir_map
from ftir_map import Project, load, open_project, read_omnic, read_unpacked, unpack
from ftir_map.omnic import OmnicFormatError, OmnicMap
from ftir_map.project import check_catalog, find_sources, safe_name, unique_name
from ftir_map.unpacked import check_map_json

from .omnic_writer import marker_bmp, write_map
from .test_exports import REAL_MAPS, grid_map, point_map
from .test_omnic import DATA, GRID, LINE, _library, spectra

RECT = [100.0, 300.0, 500.0, 600.0]


def real_path(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not in tests/data")
    return path


def messy_titles(n):
    return ["=SUM(A1)", "Pixel Ü é", *[f"Spectrum {i}" for i in range(n - 2)]]


def nan_grid(tmp_path):
    values = spectra(6)
    values[1, 3], values[4, :] = np.nan, np.inf
    path = tmp_path / "nan.map"
    write_map(path, values, GRID, kind="area", titles=messy_titles(6), history="line one\nline two")
    return read_omnic(path)


SYNTHETIC = {
    "grid-extended": lambda p: grid_map(p, images=[RECT], image_bmp=marker_bmp()),
    "grid-classic": lambda p: grid_map(p, "classic", z=[1.0] * 6),
    "points-classic-z": lambda p: point_map(p, layout="classic", z=[1.0, 2.0, 3.0, 4.0]),
    "line": lambda p: point_map(p, LINE, images=[RECT, [0.0, 0.0, 50.0, 40.0]]),
    "array": lambda p: grid_map(p, titles=[f"Pixel {i}" for i in range(6)]),
    "interferogram": lambda p: point_map(p, LINE, data_kind="interferogram"),
    "nan-and-inf": nan_grid,
}


@pytest.fixture(params=SYNTHETIC)
def synthetic(request, tmp_path):
    return SYNTHETIC[request.param](tmp_path)


def as_repr(value):
    """repr of a dataclass or None; unlike ==, it treats NaN as equal to NaN."""
    return repr(dataclasses.asdict(value)) if value is not None else "None"


def assert_same_map(a: OmnicMap, b: OmnicMap, *, axis_exact=True):
    """Every data field of the two maps is equal; spectra bit for bit (NaN equals NaN)."""
    assert a.path == b.path
    for name in ("layout", "axis_unit", "map_kind", "grid_shape", "data_kind", "history", "title", "file_time"):
        assert getattr(a, name) == getattr(b, name), name
    assert a.spectra.dtype == b.spectra.dtype == np.float32
    np.testing.assert_array_equal(a.spectra.view(np.uint32), b.spectra.view(np.uint32))
    (np.testing.assert_array_equal if axis_exact else np.testing.assert_allclose)(a.axis, b.axis)
    for name in ("x", "y", "z", "times_s", "background_index"):
        left, right = getattr(a, name), getattr(b, name)
        assert (left is None) == (right is None), name
        if left is not None:
            np.testing.assert_array_equal(left, right, err_msg=name)
            assert left.dtype == right.dtype, name
    assert (a.grid_index is None) == (b.grid_index is None)
    if a.grid_index is not None and b.grid_index is not None:
        for left, right in zip(a.grid_index, b.grid_index, strict=True):
            np.testing.assert_array_equal(left, right)
            assert left.dtype == right.dtype
    assert a.titles == b.titles
    for name in ("map_definition", "map_description"):
        assert as_repr(getattr(a, name)) == as_repr(getattr(b, name)), name
    assert_same_collection(a, b)
    assert_same_backgrounds(a, b)
    assert [(i.index, i.format, i.data, i.width, i.height, i.extent) for i in a.images] == [
        (i.index, i.format, i.data, i.width, i.height, i.extent) for i in b.images
    ]


def assert_same_collection(a: OmnicMap, b: OmnicMap):
    left, right = (dataclasses.asdict(m.collection) for m in (a, b))
    left.pop("raw"), right.pop("raw")
    assert repr(left) == repr(right)
    assert not b.collection.raw and b.blocks == []


def assert_same_backgrounds(a: OmnicMap, b: OmnicMap):
    assert len(a.backgrounds) == len(b.backgrounds)
    for left, right in zip(a.backgrounds, b.backgrounds, strict=True):
        np.testing.assert_array_equal(left.spectrum.view(np.uint32), right.spectrum.view(np.uint32))
        np.testing.assert_array_equal(left.wavenumber, right.wavenumber)
        assert (left.data_kind, left.title, left.time) == (right.data_kind, right.title, right.time)


# ------------------------------------------------------------------ round trip
def test_round_trip_synthetic(synthetic, tmp_path):
    synthetic.unpack(tmp_path / "out")
    assert_same_map(synthetic, read_unpacked(tmp_path / "out"))


@pytest.mark.parametrize("name", REAL_MAPS)
def test_round_trip_real_files(name, tmp_path):
    m = read_omnic(real_path(name))
    m.unpack(tmp_path / "out")
    assert_same_map(m, read_unpacked(tmp_path / "out"))


@pytest.mark.parametrize("spectra_format", ["zarr", "csv", "nc"])
@pytest.mark.parametrize("case", ["grid-extended", "nan-and-inf", "points-classic-z", "interferogram"])
def test_round_trip_each_spectra_format(case, spectra_format, tmp_path):
    if spectra_format == "nc":
        pytest.importorskip("h5netcdf")
    m = SYNTHETIC[case](tmp_path)
    m.unpack(tmp_path / "out", spectra=spectra_format)
    # the csv axis keeps 9 significant digits; everything else is exact
    assert_same_map(m, read_unpacked(tmp_path / "out"), axis_exact=spectra_format != "csv")


def test_csv_titles_that_look_like_formulas_come_back(tmp_path):
    m = nan_grid(tmp_path)
    m.unpack(tmp_path / "out", spectra="csv")
    assert read_unpacked(tmp_path / "out").titles[0] == "=SUM(A1)"


def test_unpacked_map_keeps_working_methods(tmp_path):
    m = grid_map(tmp_path)
    m.unpack(tmp_path / "out")
    back = read_unpacked(tmp_path / "out")
    xr.testing.assert_identical(back.to_xarray(), m.to_xarray())
    np.testing.assert_array_equal(back.spectra_grid(), m.spectra_grid())
    assert back.n_points == 6


# ------------------------------------------------------------------ open formats
def test_folder_layout_and_original_images(tmp_path):
    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    m.unpack(tmp_path / "out", png=True, copy_source=True)
    out = tmp_path / "out"
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert (out / "spectra.zarr").is_dir()  # a Zarr store is a folder of chunk files
    names = {n for n in names if not n.startswith("spectra.zarr/")}
    assert names == {
        "map.json",
        "README.txt",
        "points.csv",
        "source/grid.map",
        "images/image_01.bmp",
        "images/image_01.bpw",
        "images/images.csv",
        "images/images.json",
        "images/png/image_01.png",
        "images/png/image_01.pgw",
        "images/png/images.csv",
        "images/png/images.json",
    }
    assert (out / "images/image_01.bmp").read_bytes() == m.images[0].data
    assert (out / "source/grid.map").read_bytes() == m.path.read_bytes()
    meta = json.loads((out / "map.json").read_text(encoding="utf-8"))
    assert (meta["format"], meta["format_version"]) == ("ftir-map unpacked", 1)
    assert meta["source"]["file_name"] == "grid.map"
    assert meta["source"]["sha256"] == hashlib.sha256(m.path.read_bytes()).hexdigest()  # the file's own checksum
    assert meta["source"]["size_bytes"] == m.path.stat().st_size
    assert meta["metadata"]["collection"]["n_sample_scans"] == 16
    assert "xr.open_zarr" in (out / "README.txt").read_text(encoding="utf-8")


def test_zarr_opens_without_this_package(tmp_path):
    m = grid_map(tmp_path)
    m.unpack(tmp_path / "out")
    assert not (tmp_path / "out" / "spectra.nc").exists()
    with xr.open_zarr(tmp_path / "out" / "spectra.zarr", decode_times=False, decode_timedelta=False) as ds:
        assert ds["spectra"].dims == ("point", "wavenumber")  # plain xarray: no ftir_map call
        np.testing.assert_array_equal(ds["spectra"].values, m.spectra)
        assert ds["wavenumber"].attrs["units"] == "cm-1"
        assert ds["x"].attrs["units"] == "um"
        assert ds["time_s"].dtype == np.float64
        assert list(ds["title"].values) == m.titles
        assert set(ds.data_vars) >= {"x", "y", "grid_row", "grid_column", "background_index", "backgrounds"}
        assert ds.attrs["format"] == "ftir-map unpacked"


def test_default_unpack_needs_neither_h5netcdf_nor_h5py(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5netcdf", None)  # makes `import h5netcdf` raise ImportError
    monkeypatch.setitem(sys.modules, "h5py", None)
    m = grid_map(tmp_path)
    m.unpack(tmp_path / "out")
    assert (tmp_path / "out" / "spectra.zarr").is_dir() and not (tmp_path / "out" / "spectra.nc").exists()
    assert_same_map(m, read_unpacked(tmp_path / "out"))


def test_netcdf_is_an_opt_in_with_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5netcdf", None)
    with pytest.raises(ImportError, match=r"ftir-OMNIC-map-reader\[netcdf\]"):
        grid_map(tmp_path).unpack(tmp_path / "out", spectra="nc")


def test_netcdf_opens_without_this_package(tmp_path):
    pytest.importorskip("h5netcdf")
    h5py = pytest.importorskip("h5py")
    m = grid_map(tmp_path)
    m.unpack(tmp_path / "out", spectra="nc")
    path = tmp_path / "out" / "spectra.nc"
    with xr.open_dataset(path, engine="h5netcdf") as ds:  # plain xarray: no ftir_map call
        assert ds["spectra"].dims == ("point", "wavenumber")
        np.testing.assert_array_equal(ds["spectra"].values, m.spectra)
        assert ds["wavenumber"].attrs["units"] == "cm-1"
        assert ds["x"].attrs["units"] == "um"
        assert ds["time_s"].dtype == np.float64  # not decoded as a time delta
        assert list(ds["title"].values) == m.titles
        assert set(ds.data_vars) >= {"x", "y", "grid_row", "grid_column", "background_index", "backgrounds"}
        assert ds.attrs["format"] == "ftir-map unpacked"
    with h5py.File(path) as file:
        assert file["spectra"].shape == m.spectra.shape
        np.testing.assert_array_equal(file["spectra"][:], m.spectra)
        np.testing.assert_array_equal(file["wavenumber"][:], m.axis)
        assert file["spectra"].dtype == np.float32


# ------------------------------------------------------------------ one folder
def test_unpack_refuses_a_nonempty_folder_and_overwrite_replaces_only_its_files(tmp_path):
    m = grid_map(tmp_path)
    out = tmp_path / "out"
    m.unpack(out)
    (out / "mine.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        m.unpack(out)
    m.unpack(out, spectra="csv", overwrite=True)
    assert not (out / "spectra.zarr").exists() and (out / "spectra.csv").exists()
    assert (out / "mine.txt").read_text() == "keep"
    assert_same_map(m, read_unpacked(out), axis_exact=False)


def test_unpack_rejects_an_unknown_spectra_format(tmp_path):
    with pytest.raises(ValueError, match="npz"):
        grid_map(tmp_path).unpack(tmp_path / "out", spectra="npz")


def test_read_unpacked_rejects_other_folders(tmp_path):
    with pytest.raises(OmnicFormatError, match="map.json"):
        read_unpacked(tmp_path)
    (tmp_path / "map.json").write_text(json.dumps({"format": "ftir-map unpacked", "format_version": 2}))
    with pytest.raises(OmnicFormatError, match="version 2"):
        read_unpacked(tmp_path)


# ------------------------------------------------------------------ project
@pytest.fixture
def library(tmp_path):
    """Two maps with the same file name, a mosaic-only map, a file that is not a map, and a non-.map file."""
    root = tmp_path / "library"
    for folder in ("a", "b/deep"):
        (root / folder).mkdir(parents=True)
    write_map(root / "a" / "my map.map", spectra(6), GRID, kind="area")
    write_map(root / "b/deep" / "my map.map", spectra(4), LINE[:4], layout="classic")
    write_map(root / "a" / "mosaic only.map", spectra(3), LINE[:3], with_spectra=False)
    (root / "b" / "broken.map").write_bytes(b"not an omnic file at all" * 20)
    (root / "a" / "notes.txt").write_text("ignored")
    return root


def record(actions):
    return lambda row, action: actions.append((row["name"], action))


def test_project_with_good_and_unreadable_maps(library, tmp_path):
    project_dir = tmp_path / "project"
    actions = []
    project = unpack([library], project_dir, progress=record(actions))
    assert sorted(actions) == [
        ("broken", "skipped"),
        ("mosaic_only", "skipped"),
        ("my_map", "unpacked"),
        ("my_map_" + project.rows[3]["sha256"][:8], "unpacked"),
    ]
    assert isinstance(project, Project) and len(project) == 2 and len(project.rows) == 4
    by_name = {row["name"]: row for row in project.rows}
    assert by_name["broken"]["status"].startswith("skipped: ") and by_name["broken"]["folder"] is None
    assert "no spectra" in by_name["mosaic_only"]["status"]
    good = by_name["my_map"]
    assert (good["map_kind"], good["n_points"], good["axis_unit"], good["n_images"]) == ("grid", 6, "cm-1", 0)
    assert (good["x_min"], good["x_max"], good["y_min"], good["y_max"]) == (100.0, 150.0, 200.0, 250.0)
    assert good["folder"] == "maps/my_map" and (project_dir / good["folder"] / "map.json").is_file()
    assert not list((project_dir / "maps").glob("broken*"))
    for row in project.rows:
        if row["status"] == "ok":
            assert_same_map(read_omnic(row["source_path"]), project.load(row["name"]))
    assert list(project) == project.names and "my_map" in project
    assert project.to_xarray("my_map")["spectra"].dims == ("y", "x", "wavenumber")
    with pytest.raises(KeyError, match="mosaic_only"):
        project.load("mosaic_only")
    frame = project.catalog
    assert list(frame["name"]) == [row["name"] for row in project.rows] and len(frame) == 4


def test_catalog_files(library, tmp_path):
    unpack([library], tmp_path / "project")
    with open(tmp_path / "project" / "catalog.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["name"] for row in rows[:3]] == ["mosaic_only", "my_map", "broken"]
    assert rows[0]["status"].startswith("skipped: ") and rows[0]["n_points"] == ""
    content = json.loads((tmp_path / "project" / "catalog.json").read_text(encoding="utf-8"))
    assert content["format"] == "ftir-map project" and len(content["maps"]) == 4
    assert open_project(tmp_path / "project").rows == content["maps"]


def test_rerun_is_idempotent_and_overwrite_unpacks_again(library, tmp_path):
    project_dir = tmp_path / "project"
    unpack([library], project_dir)
    map_json = project_dir / "maps/my_map/map.json"
    before = (map_json.read_bytes(), (project_dir / "catalog.json").read_text(encoding="utf-8"))
    actions = []
    unpack([library], project_dir, progress=record(actions))
    assert sorted(action for _, action in actions) == ["skipped", "skipped", "unchanged", "unchanged"]
    assert map_json.read_bytes() == before[0]
    stale = project_dir / "maps/my_map/stale.txt"
    stale.write_text("x")
    (project_dir / "maps/my_map/spectra.zarr/damaged.txt").write_bytes(b"damaged")
    actions.clear()
    unpack([library], project_dir, overwrite=True, progress=record(actions))
    assert sorted(action for _, action in actions) == ["skipped", "skipped", "unpacked", "unpacked"]
    assert_same_map(read_omnic(library / "a" / "my map.map"), open_project(project_dir).load("my_map"))
    assert stale.exists()  # only the files the unpacker writes are replaced


def test_changed_source_and_incomplete_folder_are_unpacked_again(library, tmp_path):
    project_dir = tmp_path / "project"
    unpack([library], project_dir)
    write_map(library / "a" / "my map.map", spectra(6) + 1, GRID, kind="area")
    actions = []
    project = unpack([library], project_dir, progress=record(actions))
    assert ("my_map", "unpacked") in actions and len(project) == 2
    np.testing.assert_array_equal(project.load("my_map").spectra, spectra(6) + 1)
    (project_dir / "maps/my_map/map.json").unlink()  # an interrupted write has no map.json
    actions.clear()
    unpack([library], project_dir, progress=record(actions))
    assert ("my_map", "unpacked") in actions


def test_second_run_adds_to_the_catalog_and_copied_sources_are_not_found_again(library, tmp_path):
    project_dir = library / "project_inside"  # the project lives inside the folder that is searched
    unpack([library / "a" / "my map.map"], project_dir, copy_source=True)
    assert (project_dir / "maps/my_map/source/my map.map").is_file()
    project = unpack([library], project_dir, copy_source=True)
    assert len(project.rows) == 4 and len(project) == 2  # the copy under source/ is not a fifth map
    assert project.names[0] == "my_map"


def test_unpack_accepts_a_single_file_and_rejects_a_missing_source(library, tmp_path):
    project = unpack([library / "a" / "my map.map"], tmp_path / "project")
    assert project.names == ["my_map"]
    with pytest.raises(FileNotFoundError):
        unpack([tmp_path / "missing"], tmp_path / "other")


def test_load_dispatches_on_the_path(library, tmp_path):
    project = unpack([library], tmp_path / "project")
    assert isinstance(load(library / "a" / "my map.map"), OmnicMap)
    assert isinstance(load(tmp_path / "project" / "maps" / "my_map"), OmnicMap)
    assert isinstance(load(tmp_path / "project"), Project)
    assert ftir_map.load is load
    assert load(tmp_path / "project").rows == project.rows  # ty: ignore[unresolved-attribute]
    with pytest.raises(FileNotFoundError):
        load(tmp_path)
    with pytest.raises(OmnicFormatError, match="catalog.json"):
        open_project(tmp_path)


# ------------------------------------------------------------------ command line
def run_cli(*args):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}  # the copy under test
    return subprocess.run(
        [sys.executable, "-m", "ftir_map.cli", *args], capture_output=True, text=True, check=False, env=env
    )


def test_cli_unpack_and_catalog(library, tmp_path):
    out = str(tmp_path / "project")
    result = run_cli("unpack", str(library), "--out", out, "--png", "--copy-source")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "unpacked  my_map: 6 points, grid, 0 images" in lines
    assert "skipped   broken: not an OMNIC container (first bytes b'not an omnic file ')" in lines
    assert lines[-1] == f"2 unpacked, 0 unchanged, 2 skipped -> {out}"
    again = run_cli("unpack", str(library), "--out", out)
    assert again.stdout.splitlines()[-1] == f"0 unpacked, 2 unchanged, 2 skipped -> {out}"
    catalog = run_cli("catalog", out)
    assert catalog.returncode == 0, catalog.stderr
    header, *rows = catalog.stdout.splitlines()
    assert header.split() == ["name", "map_kind", "n_points", "axis", "data_kind", "n_images", "status"]
    assert any(row.split()[:3] == ["my_map", "grid", "6"] for row in rows) and len(rows) == 4
    assert open_project(out).names[0] == "my_map"


def test_cli_unpack_spectra_option_and_errors(library, tmp_path):
    out = tmp_path / "project"
    assert run_cli("unpack", str(library / "a" / "my map.map"), "--out", str(out)).returncode == 0
    assert (out / "maps/my_map/spectra.zarr").exists() and not (out / "maps/my_map/spectra.nc").exists()
    out_csv = tmp_path / "project_csv"
    assert (
        run_cli("unpack", str(library / "a" / "my map.map"), "--out", str(out_csv), "--spectra", "csv").returncode == 0
    )
    assert (out_csv / "maps/my_map/spectra.csv").exists()
    missing = run_cli("unpack", str(tmp_path / "nothing"), "--out", str(out))
    assert missing.returncode == 1 and "does not exist" in missing.stderr
    assert run_cli("catalog", str(tmp_path)).returncode == 1
    assert run_cli("unpack", str(library), "--out", str(out), "--spectra", "npz").returncode == 2


# ------------------------------------------------------------------ real files
def test_project_from_the_test_files(tmp_path):
    sources = [real_path(name) for name in REAL_MAPS]
    start = time.perf_counter()
    project = unpack(sources, tmp_path / "project")
    assert len(project) == len(REAL_MAPS) and time.perf_counter() - start < 120
    for source in sources:
        assert_same_map(read_omnic(source), project.load(project.names[sources.index(source)]))


@pytest.mark.skipif(not _library(), reason="set FTIR_OMNIC_MAP_DIRS to sweep a local folder of .map files")
def test_library_sweep_project(tmp_path):
    files = _library()
    project = unpack(files, tmp_path / "project")
    assert len(project.rows) == len({f.resolve() for f in files})
    for row in project.rows:
        if row["status"] == "ok":
            assert_same_map(read_omnic(row["source_path"]), project.load(row["name"]))
        else:
            assert "no spectra" in row["status"]
    assert all(row["n_points"] == project.load(row["name"]).n_points for row in project.rows if row["status"] == "ok")


# ------------------------------------------------------------------ untrusted names in map.json and catalog.json
def test_contained_refuses_paths_outside_the_base(tmp_path):
    from ftir_map.unpacked import contained

    assert contained(tmp_path, "images/image_01.bmp") == (tmp_path / "images/image_01.bmp").resolve()
    for bad in ("../outside.nc", "images/../../outside.nc", str(tmp_path.parent / "abs.nc")):
        with pytest.raises(ValueError, match="points outside"):
            contained(tmp_path, bad)


def test_read_unpacked_refuses_a_tampered_map_json(tmp_path):
    folder = tmp_path / "unpacked"
    grid_map(tmp_path).unpack(folder)
    (tmp_path / "secret.nc").write_bytes(b"not yours")
    meta_path = folder / "map.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["files"]["spectra"] = "../secret.nc"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="points outside"):
        read_unpacked(folder)


def test_project_rerun_never_touches_a_folder_named_by_a_tampered_catalog(library, tmp_path):
    project_dir = tmp_path / "project"
    unpack([library], project_dir)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "map.json").write_text("{}")  # would be cleared by an overwrite if the name were trusted
    catalog_path = project_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    row = next(r for r in catalog["maps"] if r["name"] == "my_map")
    row["name"], row["folder"] = "../../victim", "../victim"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="points outside"):
        open_project(project_dir).load("../../victim")
    unpack([library], project_dir, overwrite=True)
    assert (victim / "map.json").read_text() == "{}"
    assert all(".." not in r["name"] for r in open_project(project_dir).rows)


def test_project_refuses_to_write_through_a_symlinked_maps_folder(library, tmp_path):
    project_dir = tmp_path / "project"
    unpack([library], project_dir)
    elsewhere = tmp_path / "elsewhere"
    (project_dir / "maps").rename(elsewhere)
    (project_dir / "maps").symlink_to(elsewhere, target_is_directory=True)
    before = sorted(p.relative_to(elsewhere) for p in elsewhere.rglob("*"))
    with pytest.raises(ValueError, match="symbolic link"):
        unpack([library], project_dir, overwrite=True)
    assert sorted(p.relative_to(elsewhere) for p in elsewhere.rglob("*")) == before


@pytest.mark.parametrize("dangling", [False, True])
def test_overwrite_replaces_links_inside_a_map_folder_without_following_them(tmp_path, dangling):
    folder = tmp_path / "unpacked"
    m = grid_map(tmp_path)
    m.unpack(folder)
    outside_file, outside_dir = tmp_path / "outside.zarr", tmp_path / "outside_images"
    if not dangling:
        outside_file.write_bytes(b"keep me")
        outside_dir.mkdir()
        (outside_dir / "keep.txt").write_text("keep")
    shutil.rmtree(folder / "spectra.zarr")
    (folder / "spectra.zarr").symlink_to(outside_file)
    if (folder / "images").exists():
        shutil.rmtree(folder / "images")
    (folder / "images").symlink_to(outside_dir, target_is_directory=True)
    m.unpack(folder, overwrite=True)
    assert not (folder / "spectra.zarr").is_symlink() and not (folder / "images").is_symlink()
    if not dangling:
        assert outside_file.read_bytes() == b"keep me"
        assert [p.name for p in outside_dir.iterdir()] == ["keep.txt"]
    else:
        assert not outside_file.exists() and not outside_dir.exists()
    read_unpacked(folder)


# ------------------------------------------------------------------ review fixes: names, failures, identity
def test_names_that_differ_only_in_case_get_different_folders(tmp_path):
    for folder, name in (("a", "Sample.map"), ("b", "sample.map")):
        (tmp_path / folder).mkdir()
        write_map(tmp_path / folder / name, spectra(6) + (folder == "b"), GRID, kind="area")
    project_dir = tmp_path / "project"
    actions = []
    project = unpack([tmp_path / "a", tmp_path / "b"], project_dir, progress=record(actions))
    first, second = project.names
    assert first == "Sample" and second.casefold() != first.casefold()
    assert len({p.name.casefold() for p in (project_dir / "maps").iterdir()}) == 2
    assert project.load(first).path.name == "Sample.map" and project.load(second).path.name == "sample.map"
    again = []
    unpack([tmp_path / "a", tmp_path / "b"], project_dir, progress=record(again))
    assert {action for _, action in again} == {"unchanged"}


def test_a_new_source_does_not_reuse_an_untracked_folder_in_another_case(tmp_path):
    project_dir = tmp_path / "project"
    (project_dir / "maps" / "SAMPLE").mkdir(parents=True)
    (project_dir / "maps" / "SAMPLE" / "keep.txt").write_text("keep")
    write_map(tmp_path / "sample.map", spectra(6), GRID, kind="area")
    project = unpack([tmp_path / "sample.map"], project_dir)
    assert project.names[0].startswith("sample_") and (project_dir / "maps" / "SAMPLE" / "keep.txt").exists()


@pytest.mark.parametrize("name", ["CON", "nul", "Com1", "LPT9.x", "aux.tar"])
def test_safe_name_avoids_windows_device_names(name):
    result = safe_name(Path(f"{name}.map"))
    assert result.partition(".")[0].upper() not in {"CON", "NUL", "COM1", "LPT9", "AUX", "PRN"}
    assert result.replace("_", "", 1) == name
    assert safe_name(Path("console.map")) == "console"


def undecodable_library(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    write_map(root / "a_good.map", spectra(6), GRID, kind="area")
    write_map(root / "b_bad.map", spectra(6), GRID, kind="area", images=[RECT], image_bmp=marker_bmp()[:40])
    write_map(root / "c_good.map", spectra(5), LINE)
    return root


def test_one_undecodable_image_does_not_stop_the_project(tmp_path):
    root = undecodable_library(tmp_path)
    with pytest.warns(UserWarning, match="cannot be decoded"):
        project = unpack([root], tmp_path / "project", png=True)
    assert (tmp_path / "project" / "catalog.json").is_file() and (tmp_path / "project" / "catalog.csv").is_file()
    assert project.names == ["a_good", "b_bad", "c_good"]  # the original bytes are kept; only the PNG copy is skipped
    assert not list((tmp_path / "project" / "maps" / "b_bad" / "images" / "png").glob("*.png"))
    assert len(project.load("b_bad").images) == 1


def test_a_map_that_fails_while_unpacking_is_a_skipped_row_and_the_catalog_is_written(tmp_path, monkeypatch):
    import ftir_map.project as project_module

    root = undecodable_library(tmp_path)
    real_unpack_map = project_module.unpack_map

    def failing(m, folder, **options):
        if m.path.name == "b_bad.map":
            raise OSError("image file is truncated")  # as Pillow raises: no errno
        return real_unpack_map(m, folder, **options)

    monkeypatch.setattr(project_module, "unpack_map", failing)
    actions = []
    project = unpack([root], tmp_path / "project", progress=record(actions))
    assert ("b_bad", "skipped") in actions and project.names == ["a_good", "c_good"]
    (bad,) = (row for row in project.rows if row["name"] == "b_bad")
    assert bad["status"] == "skipped: cannot unpack: image file is truncated"


def test_a_failing_write_stops_the_run_and_the_catalog_is_written(tmp_path, monkeypatch):
    import errno

    import ftir_map.project as project_module

    root = undecodable_library(tmp_path)
    real_unpack_map = project_module.unpack_map

    def disk_full(m, folder, **options):
        if m.path.name == "b_bad.map":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_unpack_map(m, folder, **options)

    monkeypatch.setattr(project_module, "unpack_map", disk_full)
    with pytest.raises(OSError, match="No space left"):
        unpack([root], tmp_path / "project")
    assert [row["name"] for row in open_project(tmp_path / "project").rows] == ["a_good"]


def test_the_catalog_is_written_even_when_the_run_is_interrupted(tmp_path):
    root = undecodable_library(tmp_path)

    def stop(row, action):
        if row["name"] == "b_bad":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        unpack([root], tmp_path / "project", progress=stop)
    assert [row["name"] for row in open_project(tmp_path / "project").rows] == ["a_good", "b_bad"]


def test_original_images_are_not_decoded_when_the_file_gave_their_size(tmp_path, monkeypatch):
    from ftir_map import writers

    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())

    def refuse(stage_image):
        raise AssertionError("decoded")

    monkeypatch.setattr(writers, "decode_image", refuse)
    (record_,) = m.save_images(tmp_path / "images", "original")
    assert (record_["width"], record_["height"]) == (m.images[0].width, m.images[0].height)


def test_an_undecodable_image_is_skipped_with_a_warning_when_it_must_be_decoded(tmp_path):
    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp()[:40])
    with pytest.warns(UserWarning, match="cannot be decoded"):
        assert m.save_images(tmp_path / "images", "png") == []


def test_unpack_refuses_a_symlinked_catalog_file(library, tmp_path):
    project_dir = tmp_path / "project"
    unpack([library], project_dir)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    for name in ("catalog.csv", "catalog.json"):
        (project_dir / name).unlink()
        (project_dir / name).symlink_to(victim)
        with pytest.raises(ValueError, match="symbolic link"):
            unpack([library], project_dir)
        (project_dir / name).unlink()
        (project_dir / name).write_text("{}" if name.endswith("json") else "")
    assert victim.read_text() == "precious"


def test_the_same_folder_spelled_in_another_case_is_one_source(tmp_path):
    (tmp_path / "Data").mkdir()
    write_map(tmp_path / "Data" / "x.map", spectra(6), GRID, kind="area")
    other = tmp_path / "data"
    if not other.exists():
        pytest.skip("case-sensitive file system")
    actions = []
    unpack([tmp_path / "Data"], tmp_path / "project")
    project = unpack([other], tmp_path / "project", progress=record(actions))
    assert actions == [("x", "unchanged")] and len(project.rows) == 1
    assert len(find_sources([tmp_path / "Data", other])) == 1


def test_a_symlinked_source_is_one_source(tmp_path):
    write_map(tmp_path / "x.map", spectra(6), GRID, kind="area")
    (tmp_path / "link.map").symlink_to(tmp_path / "x.map")
    project = unpack([tmp_path / "x.map", tmp_path / "link.map"], tmp_path / "project")
    assert len(project.rows) == 1


def test_json_nan_in_images_and_backgrounds_is_null(tmp_path):
    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    m.images[0].extent = (float("nan"), 1.0, 2.0, 3.0)
    m.unpack(tmp_path / "out")
    text = (tmp_path / "out" / "map.json").read_text()
    assert "NaN" not in text
    json.loads(text, parse_constant=lambda name: pytest.fail(name))
    extent = read_unpacked(tmp_path / "out").images[0].extent
    assert extent is not None and np.isnan(extent[0])


def test_netcdf_with_external_storage_is_refused(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("h5netcdf")
    from ftir_map.unpacked import UnsafeNetcdfFile

    m = grid_map(tmp_path)
    m.unpack(tmp_path / "u", spectra="nc")
    read_unpacked(tmp_path / "u")  # a file written by unpack passes the check
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"x" * 1024)
    with h5py.File(tmp_path / "u" / "spectra.nc", "r+") as file:
        file.create_dataset("leak", shape=(8,), dtype="<f4", external=[(str(secret), 0, 32)])
    with pytest.raises(UnsafeNetcdfFile, match="another file"):
        read_unpacked(tmp_path / "u")


def test_netcdf_with_an_external_link_or_virtual_dataset_is_refused(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("h5netcdf")
    from ftir_map.unpacked import UnsafeNetcdfFile

    grid_map(tmp_path).unpack(tmp_path / "u", spectra="nc")
    nc = tmp_path / "u" / "spectra.nc"
    with h5py.File(nc, "r+") as file:
        file["link"] = h5py.ExternalLink(str(tmp_path / "other.h5"), "/x")
    with pytest.raises(UnsafeNetcdfFile, match="link"):
        read_unpacked(tmp_path / "u")
    with h5py.File(nc, "r+") as file:
        del file["link"]
        layout = h5py.VirtualLayout(shape=(4,), dtype="<f4")
        layout[:] = h5py.VirtualSource(str(tmp_path / "other.h5"), "x", shape=(4,))
        file.create_virtual_dataset("virtual", layout)
    with pytest.raises(UnsafeNetcdfFile, match="another file"):
        read_unpacked(tmp_path / "u")


# ------------------------------------------------------------------ what an unpack does and does not create
def test_default_unpack_creates_no_png_and_no_source_folder(tmp_path):
    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    m.unpack(tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "images").is_dir() and not (out / "images" / "png").exists()  # unpacked.md: png only with --png
    assert not (out / "source").exists()  # unpacked.md: source only with --copy-source
    assert json.loads((out / "map.json").read_text(encoding="utf-8"))["source"]["copied_to"] is None


def test_default_project_unpack_creates_no_png_and_no_source_folder(tmp_path):
    # project.unpack has its own png/copy_source defaults (what `ftir-map unpack` uses), separate from unpack_map's
    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    project = unpack([m.path], tmp_path / "project")
    folder = project.folder(project.names[0])
    assert (folder / "images").is_dir() and not (folder / "images" / "png").exists()  # unpacked.md
    assert not (folder / "source").exists()  # unpacked.md: source only with --copy-source


# ------------------------------------------------------------------ backgrounds come back exactly (unpacked.md)
@pytest.mark.parametrize("spectra_format", ["zarr", "csv"])
@pytest.mark.parametrize("layout", ["classic", "extended"])
@pytest.mark.parametrize(("first_x", "last_x"), [(4000.0, 650.0), (650.0, 4000.0)], ids=["descending", "ascending"])
def test_background_wavenumbers_round_trip_bit_for_bit(tmp_path, layout, first_x, last_x, spectra_format):
    # linspace(4000, 650) reversed differs from linspace(650, 4000) by up to 4.5e-13; the reader builds the first
    path = tmp_path / "m.map"
    write_map(path, np.zeros((4, 4), np.float32), np.zeros((4, 2)), layout=layout, first_x=first_x, last_x=last_x)
    m = read_omnic(path)
    m.unpack(tmp_path / "out", spectra=spectra_format)
    back = read_unpacked(tmp_path / "out")
    assert m.backgrounds and len(back.backgrounds) == len(m.backgrounds)
    for before, after in zip(m.backgrounds, back.backgrounds, strict=True):
        np.testing.assert_array_equal(before.wavenumber, after.wavenumber)


@pytest.mark.parametrize("spectra_format", ["zarr", "csv"])
def test_a_background_with_other_ends_than_the_header_gets_its_own_linspace(tmp_path, spectra_format):
    # map.json keeps each background's two ends (unpacked_format.md); the header's axis is used only when they match
    path = tmp_path / "m.map"
    write_map(path, np.zeros((4, 4), np.float32), np.zeros((4, 2)), first_x=650.0, last_x=4000.0)
    m = read_omnic(path)
    own_axis = np.linspace(700.0, 3900.0, len(m.axis))
    m.backgrounds[0].wavenumber = own_axis
    m.unpack(tmp_path / "out", spectra=spectra_format)
    back = read_unpacked(tmp_path / "out")
    np.testing.assert_array_equal(back.backgrounds[0].wavenumber, own_axis)
    np.testing.assert_array_equal(back.backgrounds[-1].wavenumber, m.backgrounds[-1].wavenumber)


# ------------------------------------------------------------------ names of maps with the same file name
def test_unique_name_follows_the_documented_scheme():
    sha = "0123456789abcdef" * 4
    path = Path("some/folder/a.map")
    assert unique_name(path, sha, set()) == "a"
    assert unique_name(path, sha, {"a"}) == "a_01234567"  # unpacked_format.md: the first 8 characters of the sha256
    assert unique_name(path, sha, {"a", "a_01234567"}) == "a_01234567_2"  # then a counter
    assert unique_name(path, sha, {"a", "a_01234567", "a_01234567_2"}) == "a_01234567_3"
    assert unique_name(Path("A.map"), sha, {"a"}) == "A_01234567"  # taken names are case-folded


def test_three_files_with_one_name_get_the_documented_names(tmp_path):
    first, second = tmp_path / "x" / "a.map", tmp_path / "y" / "a.map"
    third = tmp_path / "z" / "a.map"
    for folder in (first.parent, second.parent, third.parent):
        folder.mkdir()
    write_map(first, spectra(6), GRID, kind="area")
    write_map(second, spectra(4), LINE[:4], layout="classic")
    shutil.copy(second, third)  # same content, so the same checksum as the second
    project = unpack([first, second, third], tmp_path / "project")
    digest = hashlib.sha256(second.read_bytes()).hexdigest()[:8]
    assert [row["name"] for row in project.rows] == ["a", f"a_{digest}", f"a_{digest}_2"]
    assert [row["folder"] for row in project.rows] == ["maps/a", f"maps/a_{digest}", f"maps/a_{digest}_2"]
    assert sorted(p.name for p in (tmp_path / "project/maps").iterdir()) == sorted(row["name"] for row in project.rows)


# ------------------------------------------------------------------ damaged files are OmnicFormatError, never a raw error
@pytest.fixture
def unpacked_folder(tmp_path):
    grid_map(tmp_path).unpack(tmp_path / "out")
    return tmp_path / "out"


def edit_map_json(folder, change):
    path = folder / "map.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    change(content)
    path.write_text(json.dumps(content), encoding="utf-8")


def set_n_points(content):
    content["metadata"].update(n_points=99, grid_shape=None)  # no grid, so only the spectra file can say it is wrong


def drop_background_records(content):
    content["backgrounds"] = []


def set_grid_shape(content):
    content["metadata"]["grid_shape"] = [1, 2]


def drop_metadata(content):
    del content["metadata"]


def set_wrong_format(content):
    content["format"] = "other"


@pytest.mark.parametrize(
    ("change", "pattern"),
    [
        (set_n_points, r"n_points is 99 but the spectra file holds 6 spectra"),
        (drop_background_records, r"0 background records but 1 backgrounds"),
        (set_grid_shape, r"grid_shape \[1, 2\] holds 2 points but n_points is 6"),
        (drop_metadata, r"no 'metadata'"),
        (set_wrong_format, r"format is 'other'"),
    ],
)
def test_a_tampered_map_json_is_an_omnic_format_error(unpacked_folder, change, pattern):
    edit_map_json(unpacked_folder, change)
    with pytest.raises(OmnicFormatError, match=pattern):
        read_unpacked(unpacked_folder)


@pytest.mark.parametrize(("text", "pattern"), [("{not json", "not valid JSON"), ("[]", "a list, not an object")])
def test_a_map_json_that_is_not_an_object_is_an_omnic_format_error(unpacked_folder, text, pattern):
    (unpacked_folder / "map.json").write_text(text, encoding="utf-8")
    with pytest.raises(OmnicFormatError, match=pattern):
        read_unpacked(unpacked_folder)


def test_a_points_csv_with_a_row_missing_is_an_omnic_format_error(tmp_path):
    grid_map(tmp_path).unpack(tmp_path / "out", spectra="csv")
    lines = (tmp_path / "out" / "points.csv").read_text(encoding="utf-8").splitlines()
    (tmp_path / "out" / "points.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(OmnicFormatError, match="points.csv has 5 rows but spectra.csv has 6 spectra"):
        read_unpacked(tmp_path / "out")


def test_grid_cells_outside_the_grid_shape_are_an_omnic_format_error(unpacked_folder):
    edit_map_json(unpacked_folder, lambda content: content["metadata"].update(grid_shape=[1, 6]))  # 6 points, 1 row
    with pytest.raises(OmnicFormatError, match="'grid_row' does not fit grid_shape"):
        read_unpacked(unpacked_folder)


# ------------------------------------------------------------------ the pure checks
def good_map_json(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    return json.loads(json.dumps(grid_map(tmp_path).unpack(tmp_path / "out")))


def test_check_map_json_accepts_what_unpack_writes(tmp_path):
    assert check_map_json(good_map_json(tmp_path)) is None
    assert check_map_json(json.loads((tmp_path / "out" / "map.json").read_text(encoding="utf-8"))) is None


@pytest.mark.parametrize("content", [None, [], "map.json", 3])
def test_check_map_json_refuses_anything_but_an_object(content):
    with pytest.raises(OmnicFormatError, match="not an object"):
        check_map_json(content)


def test_check_map_json_lists_every_problem_in_one_message(tmp_path):
    content = good_map_json(tmp_path)
    content["format"], content["format_version"] = "other", 2
    del content["source"], content["images"]
    content["files"]["spectra_format"] = "npz"
    content["backgrounds"] = {}
    content["metadata"]["n_points"] = "6"
    del content["metadata"]["layout"], content["metadata"]["collection"]["first_x"]
    with pytest.raises(OmnicFormatError) as error:
        check_map_json(content)
    problems = str(error.value).split("; ")
    assert problems == [
        "format is 'other', not 'ftir-map unpacked'",
        "format version 2, this reader reads 1",
        "no 'source'",
        "'backgrounds' is not a list",
        "no 'images'",
        "'files' spectra_format is not one of ['zarr', 'csv', 'nc']",
        "'metadata' has no 'layout'",
        "n_points is '6', not a positive integer",
        "'collection' has no 'first_x'",
    ]


def test_check_map_json_checks_grid_shape_against_n_points(tmp_path):
    for shape, message in (
        ([2, 3], None),
        ([1, 2], "holds 2 points but n_points is 6"),
        ([6], "not null or two positive"),
    ):
        content = good_map_json(tmp_path / f"m{len(shape)}{shape[0]}")
        content["metadata"]["grid_shape"] = shape
        if message is None:
            assert check_map_json(content) is None
        else:
            with pytest.raises(OmnicFormatError, match=message):
                check_map_json(content)
    content = good_map_json(tmp_path / "null")
    content["metadata"]["grid_shape"] = None
    assert check_map_json(content) is None  # line and point maps have no grid


def test_check_map_json_checks_the_records(tmp_path):
    content = good_map_json(tmp_path)
    content["backgrounds"] = [{"title": "b"}, 3]
    content["images"] = [{"index": 1}]
    with pytest.raises(OmnicFormatError) as error:
        check_map_json(content)
    message = str(error.value)
    assert "backgrounds[0] has no 'wavenumber_last'" in message and "backgrounds[1] is not a dict" in message
    assert "images[0] has no 'extent'" in message


def test_check_map_json_checks_the_keys_of_source_and_files(tmp_path):
    content = good_map_json(tmp_path)
    del content["source"]["sha256"], content["files"]["points"]
    with pytest.raises(OmnicFormatError) as error:
        check_map_json(content)
    assert str(error.value).split("; ") == ["'source' has no 'sha256'", "'files' has no 'points'"]


def good_catalog(tmp_path):
    (tmp_path / "lib").mkdir()
    write_map(tmp_path / "lib" / "g.map", spectra(6), GRID, kind="area")
    (tmp_path / "lib" / "broken.map").write_bytes(b"not an omnic file at all" * 20)
    unpack([tmp_path / "lib"], tmp_path / "project")
    return json.loads((tmp_path / "project" / "catalog.json").read_text(encoding="utf-8"))


def test_check_catalog_accepts_what_unpack_writes(tmp_path):
    content = good_catalog(tmp_path)
    assert {row["status"] == "ok" for row in content["maps"]} == {True, False}  # an unpacked and a skipped row
    assert check_catalog(content) is None


@pytest.mark.parametrize("content", [None, [], "catalog", 3])
def test_check_catalog_refuses_anything_but_an_object(content):
    with pytest.raises(OmnicFormatError, match="not an object"):
        check_catalog(content)


def test_check_catalog_lists_every_problem_in_one_message(tmp_path):
    content = good_catalog(tmp_path)
    ok = next(row for row in content["maps"] if row["status"] == "ok")
    skipped = next(row for row in content["maps"] if row["status"] != "ok")
    content["format_version"] = 2
    del ok["axis_min"], ok["sha256"]
    ok["n_points"] = "6"
    skipped["name"] = 5
    content["maps"].append("g")
    with pytest.raises(OmnicFormatError) as error:
        check_catalog(content)
    i, j = content["maps"].index(ok), content["maps"].index(skipped)
    by_row = {
        i: [f"maps[{i}] has no 'sha256'", f"maps[{i}] has no 'axis_min'", f"maps[{i}] 'n_points' is not an integer"],
        j: [f"maps[{j}] 'name' is not a string"],
        2: ["maps[2] is not an object"],
    }
    assert str(error.value).split("; ") == [
        "not an 'ftir-map project' catalog of version 1",
        *(problem for row in sorted(by_row) for problem in by_row[row]),
    ]


def test_check_catalog_requires_numbers_in_the_axis_range_of_an_unpacked_row(tmp_path):
    content = good_catalog(tmp_path)
    ok = next(row for row in content["maps"] if row["status"] == "ok")
    ok["axis_min"], ok["axis_max"] = "650", True
    with pytest.raises(OmnicFormatError) as error:
        check_catalog(content)
    i = content["maps"].index(ok)
    assert str(error.value).split("; ") == [
        f"maps[{i}] 'axis_min' is not a number",
        f"maps[{i}] 'axis_max' is not a number",
    ]


def test_check_catalog_requires_a_list_of_maps(tmp_path):
    content = good_catalog(tmp_path)
    maps = content.pop("maps")
    with pytest.raises(OmnicFormatError, match="'maps' is missing"):
        check_catalog(content)
    with pytest.raises(OmnicFormatError, match="'maps' is not a list"):
        check_catalog({**content, "maps": dict(enumerate(maps))})


# ------------------------------------------------------------------ a damaged catalog.json through open_project
@pytest.mark.parametrize(
    ("change", "pattern"),
    [
        (lambda content: content.pop("maps"), "'maps' is missing"),
        (lambda content: content["maps"][0].pop("axis_min"), "has no 'axis_min'"),
        (lambda content: content["maps"][0].pop("sha256"), "has no 'sha256'"),
    ],
    ids=["no maps", "no axis_min", "no sha256"],
)
def test_open_project_on_a_damaged_catalog_is_an_omnic_format_error(library, tmp_path, change, pattern):
    unpack([library], tmp_path / "project")
    path = tmp_path / "project" / "catalog.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    change(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(OmnicFormatError, match=pattern):
        open_project(tmp_path / "project")
    with pytest.raises(OmnicFormatError, match=pattern):
        unpack([library], tmp_path / "project")  # a rerun reads the same catalog


def test_open_project_on_a_catalog_that_is_not_json_is_an_omnic_format_error(tmp_path):
    (tmp_path / "catalog.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(OmnicFormatError, match="not valid JSON"):
        open_project(tmp_path)
