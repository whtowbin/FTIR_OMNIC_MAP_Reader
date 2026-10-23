"""The exports of OmnicMap: xarray, NumPy, dict, Zarr and images, on synthetic files and on the files in tests/data."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from ftir_map import read_omnic

from .omnic_writer import marker_bmp, write_map
from .test_omnic import ARRAY_MAP, GRID, LINE, N_WN, POINTS, as_stored, real, spectra

LAYOUTS = ["classic", "extended"]
REAL_MAPS = [
    "GCB_1C_GC52_CPX best points-50x50um_256avg.map",
    "GCB_1C_ol54_ultrafast map_75um aperture and step.map",
    "GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map",
    "test file GCB_1C_GC52_opx profile1_50x50um_25um step (overlapping points)_256 averages_Ifg.map",
    ARRAY_MAP.name,
]


def grid_map(tmp_path, layout="extended", **kwargs):
    path = tmp_path / "grid.map"
    write_map(path, spectra(6), GRID, layout=layout, kind="area", **kwargs)
    return read_omnic(path)


def point_map(tmp_path, positions=POINTS, layout="extended", **kwargs):
    path = tmp_path / "points.map"
    write_map(path, spectra(len(positions)), positions, layout=layout, **kwargs)
    return read_omnic(path)


def assert_datasets_equal(a: xr.Dataset, b: xr.Dataset):
    """Same variables, coordinates and attrs; NaN equals NaN."""
    xr.testing.assert_identical(a, b)


# ------------------------------------------------------------------ xarray
@pytest.mark.parametrize("layout", LAYOUTS)
def test_to_xarray_grid(tmp_path, layout):
    m = grid_map(tmp_path, layout)
    ds = m.to_xarray()
    assert ds["spectra"].dims == ("y", "x", "wavenumber")
    assert ds["spectra"].dtype == np.float32
    np.testing.assert_array_equal(ds["spectra"].values.reshape(6, N_WN), spectra(6))
    np.testing.assert_array_equal(ds["x"].values, [100.0, 125.0, 150.0])  # absolute stage um
    np.testing.assert_array_equal(ds["y"].values, [200.0, 250.0])
    np.testing.assert_allclose(ds["wavenumber"].values, np.linspace(650.0, 4000.0, N_WN), rtol=1e-6)
    assert ds["time_s"].dims == ("y", "x")
    np.testing.assert_allclose(ds["time_s"].values.ravel(), 90.0 * np.arange(6))
    assert ds["backgrounds"].dims == ("background", "wavenumber")
    np.testing.assert_allclose(ds["backgrounds"].values[0], np.linspace(1.0, 2.0, N_WN), rtol=1e-6)
    assert ds.attrs["map_kind"] == "grid" and ds.attrs["grid_shape"] == [2, 3]
    assert ds.attrs["axis_unit"] == "cm-1"


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("positions", [POINTS, LINE], ids=["points", "line"])
def test_to_xarray_points(tmp_path, layout, positions):
    z = [20500.0 + i for i in range(len(positions))] if layout == "classic" else None
    m = point_map(tmp_path, positions, layout, z=z)
    ds = m.to_xarray()
    assert ds["spectra"].dims == ("point", "wavenumber")
    p = as_stored(positions)
    np.testing.assert_array_equal(ds["x"].values, p[:, 0])
    np.testing.assert_array_equal(ds["y"].values, p[:, 1])
    assert ("z" in ds.coords) == (z is not None)
    if z is not None:
        np.testing.assert_array_equal(ds["z"].values, z)
    assert list(ds["title"].values) == [f"Spectrum {i + 1} of {len(positions)}" for i in range(len(positions))]
    np.testing.assert_allclose(ds["time_s"].values, 90.0 * np.arange(len(positions)))
    assert ds["background_index"].dtype.kind == "i"
    assert ds.attrs["map_kind"] == m.map_kind


def test_attrs_are_json_and_exclude_raw_bytes(tmp_path):
    m = grid_map(tmp_path)
    attrs = m.to_xarray().attrs
    text = json.dumps(attrs)
    assert "spectral_header" not in text and "raw" not in attrs["collection"]
    assert attrs["source_path"] == str(m.path)
    assert attrs["reader_version"]
    assert attrs["collection"]["n_sample_scans"] == 16
    assert attrs["map_definition"]["kind"] == "area" and attrs["map_definition"]["x_range"] == [100.0, 150.0]
    assert attrs["data_kind"] == "absorbance"


def test_nan_is_kept_as_stored(tmp_path):
    data = spectra(4)
    data[1, 3] = np.nan
    path = tmp_path / "nan.map"
    write_map(path, data, POINTS)
    m = read_omnic(path)
    assert np.isnan(m.to_xarray()["spectra"].values[1, 3])
    assert np.isnan(m.to_numpy()[0][1, 3])


def test_interferogram_dimension_is_axis(tmp_path):
    path = tmp_path / "ifg.map"
    write_map(path, spectra(5), LINE, first_x=float(N_WN - 1), last_x=0.0, data_kind="interferogram")
    m = read_omnic(path)
    ds = m.to_xarray()
    assert ds["spectra"].dims == ("point", "axis")
    assert ds["axis"].attrs["units"] == "points"
    assert "axis" in m.to_numpy()[1] and "wavenumber" not in m.to_numpy()[1]


def test_incomplete_grid_is_exported_per_point(tmp_path):
    """A 3 x 3 grid with one spectrum missing is not a grid, so it is exported per point."""
    positions = [[100.0 + 25.0 * (i % 3), 200.0 + 50.0 * (i // 3)] for i in range(8)]
    m = point_map(tmp_path, positions)
    assert m.grid_shape is None and m.to_xarray()["spectra"].dims == ("point", "wavenumber")


# ------------------------------------------------------------------ NumPy and dict
def test_to_numpy_grid(tmp_path):
    m = grid_map(tmp_path)
    data, axes, positions, metadata = m.to_numpy()
    assert data.shape == (2, 3, N_WN)
    assert set(axes) == {"wavenumber", "x", "y"}
    np.testing.assert_array_equal(axes["x"], [100.0, 125.0, 150.0])
    assert set(positions) == {"x", "y", "grid_index"}
    assert positions["grid_index"].shape == (6, 2)
    np.testing.assert_array_equal(data[tuple(positions["grid_index"].T)], spectra(6))
    assert metadata == m.to_xarray().attrs


def test_to_numpy_points(tmp_path):
    m = point_map(tmp_path, z=None)
    data, axes, positions, _ = m.to_numpy()
    assert data.shape == (4, N_WN) and set(axes) == {"wavenumber"}
    assert set(positions) == {"x", "y"}


def test_to_dict(tmp_path):
    m = grid_map(tmp_path, images=[[0.0, 0.0, 400.0, 300.0]])
    d = m.to_dict()
    assert set(d) == {
        "spectra",
        "axes",
        "positions",
        "metadata",
        "titles",
        "time_s",
        "background_index",
        "backgrounds",
        "images",
    }
    assert d["titles"][0] == "Spectrum 1 of 6" and d["backgrounds"].shape == (1, N_WN)
    assert d["images"][0]["array"].shape == (3, 4, 3) and d["images"][0]["extent"] == (0.0, 400.0, 0.0, 300.0)


# ------------------------------------------------------------------ Zarr
@pytest.mark.parametrize("layout", LAYOUTS)
def test_zarr_round_trip_synthetic(tmp_path, layout):
    pytest.importorskip("zarr")
    for name, m in (("grid", grid_map(tmp_path, layout)), ("points", point_map(tmp_path, LINE, layout))):
        store = tmp_path / f"{name}.zarr"
        m.to_zarr(store)
        with xr.open_zarr(store) as back:
            assert_datasets_equal(back.load(), m.to_xarray())


def test_zarr_refuses_to_overwrite_by_default(tmp_path):
    pytest.importorskip("zarr")
    m = point_map(tmp_path)
    m.to_zarr(tmp_path / "a.zarr")
    with pytest.raises(FileExistsError, match="mode='w'"):
        m.to_zarr(tmp_path / "a.zarr")
    m.to_zarr(tmp_path / "a.zarr", mode="w")


def test_zarr_missing_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "zarr", None)  # makes `import zarr` raise ImportError
    with pytest.raises(ImportError, match=r"ftir-OMNIC-map-reader\[zarr\]"):
        point_map(tmp_path).to_zarr(tmp_path / "x.zarr")


# ------------------------------------------------------------------ images
def test_images_are_upright_and_have_a_stage_extent(tmp_path):
    """The BMP is stored bottom-up; the exported array shows it as OMNIC does: first row = largest stage Y."""
    rects = [[100.0, 300.0, 500.0, 600.0]]
    m = point_map(tmp_path, images=rects, image_bmp=marker_bmp())
    (image,) = m.to_images()
    assert (image.index, image.format, image.array.shape, image.array.dtype) == (1, "BMP", (3, 4, 3), np.uint8)
    np.testing.assert_array_equal(image.array[0, 0], [255, 0, 0])  # top-left red
    np.testing.assert_array_equal(image.array[-1, -1], [0, 0, 255])  # bottom-right blue
    assert image.extent == (100.0, 500.0, 300.0, 600.0)


def test_image_extent_is_ordered_even_if_the_file_stores_it_reversed(tmp_path):
    m = point_map(tmp_path, images=[[500.0, 600.0, 100.0, 300.0]])
    assert m.to_images()[0].extent == (100.0, 500.0, 300.0, 600.0)


def test_maps_without_images_give_an_empty_list(tmp_path):
    assert point_map(tmp_path).to_images() == []


def _ground_truth_files():
    """The OMNIC screenshot with stage axes and its map, found under FTIR_OMNIC_MAP_DIRS."""
    dirs = [Path(p) for p in os.environ.get("FTIR_OMNIC_MAP_DIRS", "").split(os.pathsep) if p]
    for d in dirs:
        for shot in d.rglob("GCB-S6-profile1_.JPG"):
            maps = list(shot.parent.glob("*.map"))
            if len(maps) == 1:
                return shot, maps[0]
    return None


@pytest.mark.skipif(_ground_truth_files() is None, reason="needs the GCB Scoria grain 6 profile 1 screenshot")
def test_image_orientation_against_omnic_screenshot():
    """The exported array matches OMNIC's own display (screenshot with stage axes) with no flip or rotation."""
    shot_path, map_path = _ground_truth_files()
    shot = np.asarray(Image.open(shot_path).convert("L"), dtype=float)
    (image,) = read_omnic(map_path).to_images()
    assert image.extent == pytest.approx((774.87, 1284.87, 15133.76, 15537.76), abs=0.01)  # the screenshot's axes
    gray = Image.fromarray(image.array).convert("L")
    flips = {
        "none": gray,
        "left-right": gray.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "top-bottom": gray.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        "both": gray.transpose(Image.Transpose.ROTATE_180),
    }
    score = {}
    for name, candidate in flips.items():
        plot_sized = np.asarray(candidate.resize((624, 495)), dtype=float)  # the screenshot's plot area, in pixels
        score[name] = max(
            np.corrcoef(shot[dy : dy + 495, dx : dx + 624].ravel(), plot_sized.ravel())[0, 1]
            for dy in range(6)
            for dx in range(86, 96)
        )
    assert score["none"] > 0.95
    assert score["none"] > max(score["left-right"], score["top-bottom"], score["both"]) + 0.1


# ------------------------------------------------------------------ real files
@pytest.mark.parametrize("name", REAL_MAPS)
def test_real_files_export_and_round_trip(tmp_path, name):
    m = read_omnic(real(name))
    ds = m.to_xarray()
    spectra_values, _axes, positions, metadata = m.to_numpy()
    np.testing.assert_array_equal(ds["spectra"].values, spectra_values)
    assert metadata == ds.attrs and json.dumps(metadata)
    assert np.isfinite(positions["x"]).all()
    if m.grid_shape is None:
        np.testing.assert_array_equal(ds["spectra"].values, m.spectra)
    else:
        assert m.grid_index is not None
        np.testing.assert_array_equal(ds["spectra"].values[tuple(m.grid_index)], m.spectra)
        assert ds["x"].size == m.grid_shape[1] and ds["y"].size == m.grid_shape[0]
    assert len(m.to_images()) == len(m.images)
    pytest.importorskip("zarr")
    store = tmp_path / "map.zarr"
    m.to_zarr(store)
    with xr.open_zarr(store) as back:
        assert_datasets_equal(back.load(), ds)


def test_real_images_have_extents_and_decode():
    m = read_omnic(real("GCB_1C_ol54_ultrafast map_75um aperture and step.map"))
    images = m.to_images()
    assert len(images) == 6 and {i.format for i in images} == {"JPEG"}
    for image in images:
        xmin, xmax, ymin, ymax = image.extent
        assert xmin < xmax and ymin < ymax
        assert image.array.shape[:2] == (686, 878)
        assert (xmax - xmin) / image.array.shape[1] == pytest.approx(0.58, abs=0.01)  # um per pixel


def test_real_array_map_has_16_backgrounds():
    ds = read_omnic(real(ARRAY_MAP.name)).to_xarray()
    assert ds["backgrounds"].shape == (16, ds["wavenumber"].size)
    assert ds["background_index"].dims == ("y", "x") and int(ds["background_index"].max()) == 15
