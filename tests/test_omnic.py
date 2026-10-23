"""ftir_map.omnic.read_omnic on synthetic files (both container layouts) and on the real files in tests/data."""

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ftir_map.omnic import OmnicFormatError, grid_layout, read_omnic

from .omnic_writer import write_map

DATA = Path(__file__).parent / "data"
ARRAY_MAP = DATA / "921005-6-plaig2-1p-array-1.map"
N_WN = 37
POINTS = [[556.36, 17357.45], [281.99, 17492.40], [291.36, 17130.62], [429.38, 17150.00]]
LINE = [[411.15 + 11.9 * i, 17109.14 + 21.92 * i] for i in range(5)]
GRID = [[100.0 + 25.0 * (i % 3), 200.0 + 50.0 * (i // 3)] for i in range(6)]
LAYOUTS = ["classic", "extended"]


def spectra(n):
    return (np.arange(n * N_WN, dtype=np.float32) / 100.0).reshape(n, N_WN)


def as_stored(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


# ------------------------------------------------------------------ synthetic files
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    ("positions", "kind", "expected"),
    [(POINTS, "points", "points"), (LINE, "line", "line"), (GRID, "area", "grid"), (LINE, None, "line")],
    ids=["points", "line", "grid", "line-without-definition"],
)
def test_layouts_and_map_kinds(tmp_path, layout, positions, kind, expected):
    path = tmp_path / "map.map"
    write_map(path, spectra(len(positions)), positions, layout=layout, kind=kind)
    m = read_omnic(path)
    assert m.layout == layout
    assert m.map_kind == expected
    np.testing.assert_array_equal(m.spectra, spectra(len(positions)))
    np.testing.assert_array_equal(m.x, as_stored(positions)[:, 0])
    np.testing.assert_array_equal(m.y, as_stored(positions)[:, 1])
    np.testing.assert_allclose(m.axis, np.linspace(650.0, 4000.0, N_WN), rtol=1e-6)
    np.testing.assert_array_equal(m.axis, m.wavenumber)
    assert m.axis_unit == "cm-1"
    assert m.titles[0] == f"Spectrum 1 of {len(positions)}"
    np.testing.assert_allclose(m.times_s, 90.0 * np.arange(len(positions)))
    assert m.data_kind == "absorbance"
    assert m.collection.n_sample_scans == 16
    assert m.collection.mirror_velocity == pytest.approx(2.5317, abs=1e-4)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grid_shape_and_cube(tmp_path, layout):
    path = tmp_path / "grid.map"
    write_map(path, spectra(6), GRID, layout=layout, kind="area")
    m = read_omnic(path)
    assert m.grid_shape == (2, 3)
    cube = m.spectra_grid()
    np.testing.assert_array_equal(cube[1, 2], spectra(6)[5])
    assert m.map_definition is not None
    assert m.map_definition.kind == "area"
    assert m.map_definition.title == "synthetic map"


def test_point_and_line_maps_have_no_grid(tmp_path):
    path = tmp_path / "points.map"
    write_map(path, spectra(4), POINTS, kind="points")
    m = read_omnic(path)
    assert m.grid_shape is None
    with pytest.raises(ValueError, match="no regular grid"):
        m.spectra_grid()


def test_array_detector_titles(tmp_path):
    titles = [f"Pixel {15 - i}, XPos=-1762.21, YPos= 978.78, X={p[0]:.2f}, Y={p[1]:.2f}" for i, p in enumerate(GRID)]
    path = tmp_path / "array.map"
    write_map(path, spectra(6), GRID, titles=titles)
    m = read_omnic(path)
    assert m.map_kind == "array"
    assert m.grid_shape == (2, 3)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_backgrounds_images_and_history(tmp_path, layout):
    rects = [[0.0, 0.0, 400.0, 300.0], [0.0, 300.0, 400.0, 600.0]]
    path = tmp_path / "map.map"
    write_map(path, spectra(4), POINTS, layout=layout, images=rects, history="Converted to absorbance")
    m = read_omnic(path)
    (bkg,) = m.backgrounds
    assert bkg.data_kind == "single_beam"
    np.testing.assert_allclose(bkg.spectrum, np.linspace(1.0, 2.0, N_WN), rtol=1e-6)
    if layout == "extended":
        assert bkg.title == "background 1"
        assert bkg.time == dt.datetime(2017, 1, 11, 14, 43, 28)
    assert [im.format for im in m.images] == ["BMP", "BMP"]
    assert (m.images[1].width, m.images[1].height) == (4, 3)
    assert m.images[1].extent == (0.0, 300.0, 400.0, 600.0)
    assert m.images[1].pixel_size_um == (100.0, 100.0)
    assert m.history.startswith("Converted to absorbance")
    assert m.file_time == dt.datetime(2017, 1, 11, 15)


def test_interferogram_is_flagged(tmp_path):
    path = tmp_path / "ifg.map"
    write_map(path, spectra(5), LINE, first_x=float(N_WN - 1), last_x=0.0, data_kind="interferogram")
    m = read_omnic(path)
    assert m.data_kind == "interferogram"
    assert m.collection.x_units == "data points"
    assert m.axis[0] == N_WN - 1  # stored order kept: the axis is the point index, not a wavenumber
    assert m.axis_unit == "points"
    with pytest.raises(ValueError, match="not cm-1"):
        _ = m.wavenumber


@pytest.mark.parametrize("layout", LAYOUTS)
def test_map_without_spectra_fails_loudly(tmp_path, layout):
    path = tmp_path / "mosaic_only.map"
    write_map(path, spectra(4), POINTS, layout=layout, with_spectra=False)
    with pytest.raises(OmnicFormatError, match="mosaic_only.map: no spectra"):
        read_omnic(path)


def test_not_an_omnic_file(tmp_path):
    path = tmp_path / "x.map"
    path.write_bytes(b"6.749918e+002,1.54\n" * 40)
    with pytest.raises(OmnicFormatError, match="not an OMNIC container"):
        read_omnic(path)


def test_grid_layout_rejects_incomplete_grids():
    x, y = np.array([0.0, 25.0, 50.0, 0.0, 25.0]), np.array([0.0, 0.0, 0.0, 50.0, 50.0])
    assert grid_layout(x, y) is None
    assert grid_layout(np.append(x, 50.0), np.append(y, 50.0))[0] == (2, 3)


# ------------------------------------------------------------------ real files (tests/data, see tests/data/README.md)
def real(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not in tests/data")
    return path


def test_real_point_map_matches_omnic_csv_export():
    m = read_omnic(real("GCB_1C_GC52_CPX best points-50x50um_256avg.map"))
    assert (m.layout, m.map_kind, m.spectra.shape) == ("classic", "points", (6, 1725))
    csvs = sorted((DATA / "cpx52_best_points_csv").glob("*avg000[1-6].CSV"))
    assert len(csvs) == 6
    for i, csv in enumerate(csvs):
        exported = np.loadtxt(csv, delimiter=",")
        np.testing.assert_allclose(m.axis, exported[:, 0], atol=1e-3)
        np.testing.assert_allclose(m.spectra[i], exported[:, 1], atol=5.01e-7)  # the CSV keeps 7 significant digits
    bkg = np.loadtxt(
        DATA / "cpx52_best_points_csv" / "GCB_1C_GC52_CPX best points-50x50um_256avgbkg1.CSV", delimiter=","
    )
    np.testing.assert_allclose(m.backgrounds[0].spectrum, bkg[:, 1], atol=5.01e-7)
    np.testing.assert_allclose(m.x, [556.36, 281.99, 291.36, 429.38, 481.05, 509.80], atol=0.01)
    assert m.z is not None
    np.testing.assert_allclose(m.z, [20507.25, 20440.5, 20193.75, 20496.0, 20484.75, 20272.5])
    assert len(m.images) == 2 and m.images[0].pixel_size_um == pytest.approx((0.5809, 0.5889), abs=1e-4)


def test_real_grid_map():
    m = read_omnic(real("GCB_1C_ol54_ultrafast map_75um aperture and step.map"))
    assert (m.layout, m.map_kind, m.grid_shape, m.spectra.shape) == ("extended", "grid", (13, 11), (143, 432))
    assert m.map_definition is not None
    assert m.map_definition.kind == "area" and (m.map_definition.n_x, m.map_definition.n_y) == (11, 13)
    assert len(m.images) == 6 and {im.format for im in m.images} == {"JPEG"}


def test_real_line_scan():
    m = read_omnic(real("GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map"))
    assert (m.map_kind, m.spectra.shape) == ("line", (21, 1725))
    np.testing.assert_allclose(np.hypot(np.diff(m.x), np.diff(m.y)), 50.0, atol=0.05)
    assert m.map_definition is not None
    assert m.map_definition.kind == "line" and m.map_definition.step == 50.0


def test_real_interferogram_map():
    m = read_omnic(
        real("test file GCB_1C_GC52_opx profile1_50x50um_25um step (overlapping points)_256 averages_Ifg.map")
    )
    assert (m.data_kind, m.map_kind, m.spectra.shape) == ("interferogram", "line", (12, 8192))
    assert m.titles[1] == "Wed Jan 11 16:34:11 2017 (GMT+00:00)"
    np.testing.assert_allclose(m.times_s[:3], [0.0, 4.0, 7.0])


def test_real_array_detector_map():
    m = read_omnic(real(ARRAY_MAP.name))
    assert (m.map_kind, m.grid_shape, m.spectra.shape) == ("array", (43, 31), (1333, 853))
    assert m.data_kind == "absorbance" and m.history.startswith("Converted to absorbance")
    title_xy = np.array([[float(t.split("X=")[1].split(",")[0]), float(t.split("Y=")[1])] for t in m.titles])
    np.testing.assert_allclose(np.column_stack([m.x, m.y]), title_xy, atol=0.006)  # titles carry 2 decimals
    assert m.background_index is not None
    assert m.background_index.max() == len(m.backgrounds) - 1 == 15


# ------------------------------------------------------------------ optional sweep over a local map library
def _library():
    dirs = [Path(p) for p in os.environ.get("FTIR_OMNIC_MAP_DIRS", "").split(os.pathsep) if p]
    return sorted(f for d in dirs for f in d.rglob("*.map"))


@pytest.mark.skipif(not _library(), reason="set FTIR_OMNIC_MAP_DIRS to sweep a local folder of .map files")
@pytest.mark.parametrize("path", _library(), ids=lambda p: p.name)
def test_library_sweep(path):
    try:
        m = read_omnic(path)
    except OmnicFormatError as e:
        assert "no spectra" in str(e)
        return
    assert np.isfinite(m.x).all() and len(m.x) == m.n_points
    assert m.map_kind in {"grid", "line", "points", "array"}
    if m.background_index is not None and m.backgrounds:
        assert m.background_index.max() < len(m.backgrounds)
    assert all(im.extent is not None for im in m.images)
    spectra, _axes, positions, metadata = m.to_numpy()  # every export works on every readable file
    assert spectra.shape[-1] == len(m.axis) and len(positions["x"]) == m.n_points
    json.dumps(metadata)
    dataset = m.to_xarray()
    json.dumps(dataset.attrs)
    np.testing.assert_array_equal(dataset["spectra"].values, spectra)
    assert len(m.to_images()) == len(m.images)


def test_package_exposes_its_version():
    import ftir_map
    from ftir_map.__about__ import __version__

    assert ftir_map.__version__ == __version__
