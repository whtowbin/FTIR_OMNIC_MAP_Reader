"""The file writers of OmnicMap: metadata, points, spectra, images with world files, export_all and the CLI."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from ftir_map import read_omnic, read_unpacked
from ftir_map.exports import build_metadata
from ftir_map.writers import csv_cell, flatten_metadata, points_table, world_file_text

from .omnic_writer import marker_bmp
from .test_exports import REAL_MAPS, grid_map, point_map
from .test_omnic import DATA

RECT = [100.0, 300.0, 500.0, 600.0]  # x0, y0, x1, y1 -> extent (100, 500, 300, 600)
MOSAIC = "GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def real_map(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not in tests/data")
    return read_omnic(path)


@pytest.fixture(params=["grid", "points", "classic-z"])
def synthetic(request, tmp_path):
    if request.param == "grid":
        return grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    if request.param == "points":
        return point_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    return point_map(tmp_path, layout="classic", z=[1.0, 2.0, 3.0, 4.0])


# ------------------------------------------------------------------ metadata
def test_flatten_metadata_uses_dotted_keys_and_indexes():
    flat = flatten_metadata({"a": {"b": 1, "c": [2, {"d": 3}]}, "e": [], "f": None})
    assert flat == {"a.b": 1, "a.c[0]": 2, "a.c[1].d": 3, "e": [], "f": None}


@pytest.mark.parametrize("name", ["json", "yaml", "yml"])
def test_metadata_json_and_yaml_round_trip(synthetic, tmp_path, name):
    if name != "json":
        pytest.importorskip("yaml")
    path = tmp_path / f"meta.{name}"
    synthetic.save_metadata(path)
    if name == "json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == build_metadata(synthetic)
    assert "raw" not in loaded["collection"]


def test_metadata_csv_has_every_flattened_key(synthetic, tmp_path):
    synthetic.save_metadata(tmp_path / "meta.json")
    synthetic.save_metadata(tmp_path / "meta.csv")
    flat = flatten_metadata(json.loads((tmp_path / "meta.json").read_text(encoding="utf-8")))
    rows = read_csv(tmp_path / "meta.csv")
    assert rows[0] == ["key", "value"]
    assert [key for key, _ in rows[1:]] == list(flat)
    values = dict(rows[1:])
    assert values["collection.n_sample_scans"] == "16"
    assert values["n_points"] == str(synthetic.n_points)


def test_metadata_explicit_format_overrides_extension(tmp_path):
    m = point_map(tmp_path)
    m.save_metadata(tmp_path / "meta.txt", format="json")
    assert json.loads((tmp_path / "meta.txt").read_text(encoding="utf-8"))["n_points"] == 4


def test_metadata_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="use one of"):
        point_map(tmp_path).save_metadata(tmp_path / "meta.xml")


def test_yaml_missing_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match=r"ftir-OMNIC-map-reader\[yaml\]"):
        point_map(tmp_path).save_metadata(tmp_path / "meta.yaml")


# ------------------------------------------------------------------ points
def test_points_csv_grid(tmp_path):
    m = grid_map(tmp_path)
    m.save_points(tmp_path / "points.csv")
    header, *rows = read_csv(tmp_path / "points.csv")
    assert header == ["index", "x_um", "y_um", "row", "column", "title", "time_s", "background_index"]
    assert len(rows) == 6
    assert rows[4][:3] == ["4", "125.0", "250.0"] and rows[4][3:5] == ["1", "1"]
    assert float(rows[2][6]) == 180.0


def test_points_columns_for_point_map_with_z(tmp_path):
    m = point_map(tmp_path, layout="classic", z=[1.0, 2.0, 3.0, 4.0])
    columns = points_table(m)
    assert "z_um" in columns and "row" not in columns
    assert columns["z_um"] == [1.0, 2.0, 3.0, 4.0]


def test_points_json_records_and_parquet_agree(synthetic, tmp_path):
    synthetic.save_points(tmp_path / "points.json")
    records = json.loads((tmp_path / "points.json").read_text(encoding="utf-8"))
    assert len(records) == synthetic.n_points and records[1]["index"] == 1
    assert records[1]["x_um"] == pytest.approx(float(synthetic.x[1]))
    parquet = pytest.importorskip("pyarrow.parquet")
    synthetic.save_points(tmp_path / "points.parquet")
    assert parquet.read_table(tmp_path / "points.parquet").to_pylist() == records


def test_parquet_missing_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"ftir-OMNIC-map-reader\[parquet\]"):
        point_map(tmp_path).save_points(tmp_path / "points.parquet")


# ------------------------------------------------------------------ spectra
def test_spectra_csv_round_trip(synthetic, tmp_path):
    synthetic.save_spectra(tmp_path / "spectra.csv")
    header = read_csv(tmp_path / "spectra.csv")[0]
    assert header == ["wavenumber_cm-1", *map(str, range(synthetic.n_points))]
    table = np.loadtxt(tmp_path / "spectra.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(table[:, 0], synthetic.axis, rtol=1e-8)
    np.testing.assert_array_equal(table[:, 1:].T.astype(np.float32), synthetic.spectra)


def test_spectra_npz_round_trip(synthetic, tmp_path):
    synthetic.save_spectra(tmp_path / "spectra.npz")
    with np.load(tmp_path / "spectra.npz", allow_pickle=False) as npz:
        np.testing.assert_array_equal(npz["spectra"], synthetic.spectra)
        np.testing.assert_array_equal(npz["axis"], synthetic.axis)
        np.testing.assert_array_equal(npz["x"], synthetic.x)
        np.testing.assert_array_equal(npz["y"], synthetic.y)
        assert ("z" in npz.files) == (synthetic.z is not None)
        assert ("grid_index" in npz.files) == (synthetic.grid_index is not None)
        assert json.loads(str(npz["metadata"])) == build_metadata(synthetic)


def test_spectra_netcdf_round_trip(synthetic, tmp_path):
    pytest.importorskip("h5netcdf")
    synthetic.save_spectra(tmp_path / "spectra.nc")
    expected = synthetic.to_xarray()
    with xr.open_dataset(tmp_path / "spectra.nc", engine="h5netcdf") as back:
        np.testing.assert_array_equal(back["spectra"].values, expected["spectra"].values)
        assert back["spectra"].dims == expected["spectra"].dims
        assert json.loads(back.attrs["metadata_json"]) == build_metadata(synthetic)


def test_spectra_zarr_round_trip_and_replace(synthetic, tmp_path):
    pytest.importorskip("zarr")
    synthetic.save_spectra(tmp_path / "spectra.zarr")
    synthetic.save_spectra(tmp_path / "spectra.zarr")  # a second write replaces the store
    with xr.open_zarr(tmp_path / "spectra.zarr") as back:
        xr.testing.assert_identical(back.load(), synthetic.to_xarray())


def test_netcdf_missing_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5netcdf", None)
    with pytest.raises(ImportError, match=r"ftir-OMNIC-map-reader\[netcdf\]"):
        point_map(tmp_path).save_spectra(tmp_path / "spectra.nc")


# ------------------------------------------------------------------ images and world files
def test_world_file_arithmetic():
    lines = world_file_text((100.0, 500.0, 300.0, 600.0), width=4, height=3).split()
    pixel_x, zero1, zero2, minus_pixel_y, x0, y0 = map(float, lines)
    assert (pixel_x, minus_pixel_y) == (100.0, -100.0) and zero1 == zero2 == 0.0
    assert (x0, y0) == (150.0, 550.0)  # upper-left pixel centre: half a pixel inside (xmin, ymax)


@pytest.mark.parametrize(
    ("format", "image_suffix", "world_suffix"),
    [("png", ".png", ".pgw"), ("tiff", ".tif", ".tfw"), ("original", ".bmp", ".bpw")],
)
def test_save_images_formats_and_world_files(tmp_path, format, image_suffix, world_suffix):
    m = point_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    folder = tmp_path / "images"
    (record,) = m.save_images(folder, format)
    image_path, world_path = folder / f"image_01{image_suffix}", folder / f"image_01{world_suffix}"
    assert record["file"] == image_path.name and record["world_file"] == world_path.name
    with Image.open(image_path) as saved:
        pixels = np.asarray(saved.convert("RGB"))
    np.testing.assert_array_equal(pixels, m.to_images()[0].array)  # upright, red top-left
    assert [float(v) for v in world_path.read_text().split()] == [100.0, 0.0, 0.0, -100.0, 150.0, 550.0]
    if format == "original":
        assert image_path.read_bytes() == m.images[0].data


def test_image_index_files(tmp_path):
    m = point_map(tmp_path, images=[RECT, [0.0, 0.0, 40.0, 30.0]], image_bmp=marker_bmp())
    m.save_images(tmp_path / "out")
    records = json.loads((tmp_path / "out" / "images.json").read_text())
    rows = read_csv(tmp_path / "out" / "images.csv")
    assert rows[0][:5] == ["file", "index", "source_format", "width", "height"]
    assert len(records) == len(rows) - 1 == 2
    first = records[0]
    assert (first["xmin"], first["xmax"], first["ymin"], first["ymax"]) == (100.0, 500.0, 300.0, 600.0)
    assert (first["width"], first["height"], first["pixel_size_x_um"], first["pixel_size_y_um"]) == (4, 3, 100.0, 100.0)
    assert first["source_format"] == "BMP" and first["orientation_verified"] is True
    assert rows[2][0] == "image_02.png"


def test_map_without_images_writes_nothing(tmp_path):
    assert point_map(tmp_path).save_images(tmp_path / "none") == []
    assert not (tmp_path / "none").exists()


def test_save_images_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="use one of"):
        point_map(tmp_path, images=[RECT]).save_images(tmp_path / "x", "gif")


def test_real_mosaic_images_original_and_png(tmp_path):
    m = real_map(MOSAIC)
    records = m.save_images(tmp_path / "jpg", "original")
    assert len(records) == 6 and {r["source_format"] for r in records} == {"JPEG"}
    assert all((tmp_path / "jpg" / r["file"]).read_bytes() == s.data for r, s in zip(records, m.images, strict=True))
    for record in records:
        world = [float(v) for v in (tmp_path / "jpg" / record["world_file"]).read_text().split()]
        assert world[4] == pytest.approx(record["xmin"] + record["pixel_size_x_um"] / 2)
        assert world[5] == pytest.approx(record["ymax"] - record["pixel_size_y_um"] / 2)
        assert world[0] * record["width"] == pytest.approx(record["xmax"] - record["xmin"])
    m.save_images(tmp_path / "png", "png")
    with Image.open(tmp_path / "png" / "image_01.png") as first:
        assert first.size == (878, 686)


# ------------------------------------------------------------------ export_all
def test_export_all_layout_and_readme(tmp_path):
    pytest.importorskip("zarr")
    m = grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp())
    folder = m.export_all(tmp_path / "out")
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["README.txt", "images", "metadata.json", "points.csv", "spectra.zarr"]
    assert sorted(p.name for p in (folder / "images").iterdir()) == [
        "image_01.pgw",
        "image_01.png",
        "images.csv",
        "images.json",
    ]
    readme = (folder / "README.txt").read_text()
    assert all(name in readme for name in ["spectra.zarr", "points.csv", "metadata.json", "images/"])
    with xr.open_zarr(folder / "spectra.zarr") as back:
        xr.testing.assert_identical(back.load(), m.to_xarray())


def test_export_all_without_images_skips_the_folder(tmp_path):
    folder = point_map(tmp_path).export_all(tmp_path / "out", spectra="npz", metadata="csv", points="json")
    assert sorted(p.name for p in folder.iterdir()) == ["README.txt", "metadata.csv", "points.json", "spectra.npz"]


def test_export_all_images_none(tmp_path):
    m = point_map(tmp_path, images=[RECT])
    folder = m.export_all(tmp_path / "out", spectra="csv", images="none")
    assert not (folder / "images").exists()


def test_export_all_refuses_a_non_empty_folder_unless_overwrite(tmp_path):
    m = point_map(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "mine.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        m.export_all(out, spectra="csv")
    assert [p.name for p in out.iterdir()] == ["mine.txt"]  # nothing written
    m.export_all(out, spectra="csv", overwrite=True)
    assert (out / "spectra.csv").exists() and (out / "mine.txt").read_text() == "keep"


def test_export_all_empty_existing_folder_is_fine(tmp_path):
    (tmp_path / "out").mkdir()
    point_map(tmp_path).export_all(tmp_path / "out", spectra="csv")


def test_export_all_checks_formats_before_writing(tmp_path):
    with pytest.raises(ValueError, match="use one of"):
        point_map(tmp_path).export_all(tmp_path / "out", metadata="xml")
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------ real files
@pytest.mark.parametrize("name", REAL_MAPS)
def test_real_maps_export_all(tmp_path, name):
    pytest.importorskip("h5netcdf")
    m = real_map(name)
    folder = m.export_all(tmp_path / "out", spectra="nc", metadata="csv", points="json")
    assert (folder / "spectra.nc").stat().st_size > 0
    assert len(json.loads((folder / "points.json").read_text())) == m.n_points
    assert (folder / "images").exists() == bool(m.images)
    m.save_spectra(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz") as npz:
        np.testing.assert_array_equal(npz["spectra"], m.spectra)  # NaN positions equal too


def test_real_grid_metadata_round_trips_json_and_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    m = real_map("GCB_1C_ol54_ultrafast map_75um aperture and step.map")
    m.save_metadata(tmp_path / "m.json")
    m.save_metadata(tmp_path / "m.yml")
    from_json = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert yaml.safe_load((tmp_path / "m.yml").read_text(encoding="utf-8")) == from_json == build_metadata(m)


# ------------------------------------------------------------------ command line
def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "ftir_map.cli", *args], capture_output=True, text=True, check=False)


def test_cli_info_on_a_real_map():
    path = DATA / "GCB_1C_ol54_ultrafast map_75um aperture and step.map"
    if not path.exists():
        pytest.skip("map not in tests/data")
    result = run_cli("info", str(path))
    assert result.returncode == 0, result.stderr
    assert "map kind: grid (13 rows x 11 columns)" in result.stdout
    assert "points: 143" in result.stdout and "images: 6" in result.stdout
    assert "n_sample_scans:" in result.stdout


def test_cli_export_on_a_real_map(tmp_path):
    path = DATA / MOSAIC
    if not path.exists():
        pytest.skip("map not in tests/data")
    out = tmp_path / "out"
    result = run_cli(
        "export", str(path), "--out", str(out), "--spectra", "npz", "--metadata", "csv", "--images", "tiff"
    )
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == [
        "README.txt",
        "images",
        "metadata.csv",
        "points.csv",
        "spectra.npz",
    ]
    assert (out / "images" / "image_01.tfw").exists()
    again = run_cli("export", str(path), "--out", str(out))
    assert again.returncode == 1 and "--overwrite" in again.stderr
    assert run_cli("export", str(path), "--out", str(out), "--spectra", "npz", "--overwrite").returncode == 0


def test_cli_reports_errors_without_a_traceback(tmp_path):
    result = run_cli("info", str(tmp_path / "missing.map"))
    assert result.returncode == 1 and result.stderr.startswith("ftir-map:") and "Traceback" not in result.stderr


def test_cli_rejects_unknown_format():
    assert run_cli("export", "x.map", "--out", "y", "--spectra", "hdf").returncode == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("=HYPERLINK(1)", "'=HYPERLINK(1)"),
        ("+1", "'+1"),
        ("-cmd", "'-cmd"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tx", "'\tx"),
        ("Spectrum 1 of 336", "Spectrum 1 of 336"),
        (-3.5, -3.5),
        (None, ""),
    ],
)
def test_csv_cell_escapes_formula_text_only(value, expected):
    assert csv_cell(value) == expected


def test_points_csv_escapes_untrusted_titles(tmp_path):
    m = grid_map(tmp_path)
    m.titles[0] = "=1+1"
    out = tmp_path / "points.csv"
    m.save_points(out)
    with out.open(newline="", encoding="utf-8") as file:
        first = next(csv.DictReader(file))
    assert first["title"] == "'=1+1"


# ------------------------------------------------------------------ review fixes
@pytest.mark.parametrize("title", ["'=x", "'+1", "''", "'", "=x", "'plain", "@a", "-5", "ok"])
def test_titles_with_quotes_and_formula_characters_round_trip_through_csv(tmp_path, title):
    m = grid_map(tmp_path)
    m.titles[0] = title
    m.unpack(tmp_path / "out", spectra="csv")
    assert read_unpacked(tmp_path / "out").titles[0] == title
    assert csv_cell(title).startswith("'") == title.startswith(("'", "=", "+", "-", "@"))


def test_export_all_with_a_missing_optional_package_writes_nothing(tmp_path, monkeypatch):
    import importlib

    real_import = importlib.import_module

    def without_pyarrow(name, *args, **kwargs):
        if name.split(".")[0] == "pyarrow":
            raise ImportError("no pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_pyarrow)
    out = tmp_path / "out"
    with pytest.raises(ImportError, match="pyarrow"):
        grid_map(tmp_path, images=[RECT], image_bmp=marker_bmp()).export_all(out, points="parquet")
    assert not out.exists()


@pytest.mark.parametrize(("kind", "package"), [("metadata", "yaml"), ("spectra", "zarr")])
def test_export_all_checks_every_chosen_format_before_writing(tmp_path, monkeypatch, kind, package):
    import importlib

    real_import = importlib.import_module

    def without(name, *args, **kwargs):
        if name == package:
            raise ImportError(f"no {package}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without)
    options = {"metadata": "yaml"} if kind == "metadata" else {"spectra": "zarr"}
    with pytest.raises(ImportError):
        grid_map(tmp_path).export_all(tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_save_spectra_zarr_refuses_an_ordinary_folder_with_files(tmp_path):
    pytest.importorskip("zarr")
    m = grid_map(tmp_path)
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "important.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not a Zarr store"):
        m.save_spectra(folder, "zarr")
    assert (folder / "important.txt").read_text() == "keep"
    m.save_spectra(tmp_path / "s.zarr", "zarr")
    m.save_spectra(tmp_path / "s.zarr", "zarr")  # an existing store is still replaced
    empty = tmp_path / "empty"
    empty.mkdir()
    m.save_spectra(empty, "zarr")  # so is an empty folder


def test_to_zarr_mode_w_refuses_an_ordinary_folder_with_files(tmp_path):
    pytest.importorskip("zarr")
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "important.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not a Zarr store"):
        grid_map(tmp_path).to_zarr(folder, mode="w")
    assert (folder / "important.txt").exists()
