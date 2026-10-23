"""Unpacked maps and project folders: round trip, open formats, idempotent re-runs, the command line."""

import csv
import dataclasses
import json
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest
import xarray as xr

import ftir_map
from ftir_map import Project, load, open_project, read_omnic, read_unpacked, unpack
from ftir_map.omnic import OmnicFormatError, OmnicMap

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


@pytest.mark.parametrize("spectra_format", ["zarr", "csv"])
@pytest.mark.parametrize("case", ["grid-extended", "nan-and-inf", "points-classic-z", "interferogram"])
def test_round_trip_zarr_and_csv(case, spectra_format, tmp_path):
    if spectra_format == "zarr":
        pytest.importorskip("zarr")
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
    assert names == {
        "map.json",
        "README.txt",
        "spectra.nc",
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
    assert meta["source"]["file_name"] == "grid.map" and len(meta["source"]["sha256"]) == 64
    assert meta["metadata"]["collection"]["n_sample_scans"] == 16
    assert "ncread" in (out / "README.txt").read_text(encoding="utf-8")


def test_netcdf_opens_without_this_package(tmp_path):
    m = grid_map(tmp_path)
    m.unpack(tmp_path / "out")
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
    pytest.importorskip("zarr")
    m.unpack(out, spectra="zarr", overwrite=True)
    assert not (out / "spectra.nc").exists() and (out / "spectra.zarr").exists()
    assert (out / "mine.txt").read_text() == "keep"
    assert_same_map(m, read_unpacked(out))


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
    (project_dir / "maps/my_map/spectra.nc").write_bytes(b"damaged")
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
    return subprocess.run([sys.executable, "-m", "ftir_map.cli", *args], capture_output=True, text=True, check=False)


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
    pytest.importorskip("zarr")
    out = tmp_path / "project"
    assert run_cli("unpack", str(library / "a" / "my map.map"), "--out", str(out), "--spectra", "zarr").returncode == 0
    assert (out / "maps/my_map/spectra.zarr").exists()
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
