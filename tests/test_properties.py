"""Property tests (Hypothesis): inverses and round trips that must hold for every input a caller can pass.

Run with HYPOTHESIS_PROFILE=ci for 200 examples per property (the default "dev" profile draws 50; see conftest.py).
Sound domains: the field sizes of docs/site/file_format.md and the promises of docs/site/map_kinds.md.
"""

import csv

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from ftir_map import read_omnic, read_unpacked
from ftir_map.omnic import grid_layout
from ftir_map.unpacked import _unescape_cell
from ftir_map.writers import csv_cell, world_file_text, write_csv

from .omnic_writer import write_map
from .test_unpacked import assert_same_map

# the tmp_path_factory fixture is session scoped, so one folder per example is safe
FILE_PROPERTY = settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])

# float32 in the file: every value, NaN, the infinities and subnormals included
float32s = st.floats(width=32, allow_nan=True, allow_infinity=True)
stage_um = st.floats(width=32, min_value=-1e5, max_value=1e5)  # stored as float32; |value| well above any stage
# title field: char[64], latin-1, no NUL (docs/site/file_format.md, spectrum header +16)
titles = st.text(st.characters(codec="latin-1", exclude_characters="\0"), max_size=64)
# (first X, last X): two ascending axes and the descending one OMNIC can store (the reader reverses it)
AXES = [(650.0, 4000.0), (674.9918, 3999.8), (4000.0, 650.0)]


@st.composite
def synthetic_maps(draw):
    """Arguments for `write_map` plus the spectra: a point list of 1-6 spectra, or a 2 x 3 lattice."""
    layout = draw(st.sampled_from(["classic", "extended"]))
    lattice = draw(st.booleans())
    n = 6 if lattice else draw(st.integers(1, 6))
    n_wn = draw(st.integers(2, 9))
    spectra = np.array(draw(st.lists(float32s, min_size=n * n_wn, max_size=n * n_wn)), dtype=np.float32)
    if lattice:
        x0, y0, dx, dy = (draw(st.floats(-1e4, 1e4)) for _ in range(4))
        dx, dy = abs(dx) + 5.0, abs(dy) + 5.0
        positions = [(x0 + dx * i, y0 + dy * j) for j in range(2) for i in range(3)]
    else:
        coordinate = (stage_um | st.just(float("nan"))) if draw(st.booleans()) else stage_um
        positions = draw(st.lists(st.tuples(coordinate, coordinate), min_size=n, max_size=n))
    first_x, last_x = draw(st.sampled_from(AXES))
    z = None
    if layout == "classic" and draw(st.booleans()):  # the stage-Z block exists in the classic layout only
        z = draw(st.lists(st.floats(width=32, min_value=0, max_value=3e4), min_size=n, max_size=n))
    return {
        "spectra": spectra.reshape(n, n_wn),
        "positions": np.array(positions, dtype=np.float32),
        "layout": layout,
        "first_x": first_x,
        "last_x": last_x,
        "titles": draw(st.lists(titles, min_size=n, max_size=n)),
        "z": z,
        # the uint32 centisecond counter wraps at 2**32; times are differences modulo 2**32 (file_format.md, +4)
        "start_centiseconds": draw(st.sampled_from([0, 8878, 2**32 - 9001, 2**32 - 1]) | st.integers(0, 2**32 - 1)),
    }


def write_spec(path, spec):
    kwargs = {k: v for k, v in spec.items() if k not in ("spectra", "positions")}
    write_map(path, spec["spectra"], spec["positions"], **kwargs)


# ------------------------------------------------------------------ writer -> reader -> unpacked
def match_csv_precision(m, back):
    """A CSV unpack keeps numbers to nine significant digits (docs/site/unpacked_format.md), so the axis and the
    backgrounds' wavenumbers are only close: relative 5e-9 (half a unit in the ninth digit). That holds a float32
    exactly, so the spectra stay bit-exact. After the check, copy them over so `assert_same_map` can test the rest."""
    np.testing.assert_allclose(back.axis, m.axis, rtol=5e-9)
    back.axis = m.axis
    for left, right in zip(m.backgrounds, back.backgrounds, strict=True):
        np.testing.assert_allclose(right.wavenumber, left.wavenumber, rtol=5e-9)
        right.wavenumber = left.wavenumber
    # a CSV cell says only "nan": the sign and payload bits of a NaN are not kept (measured: 0xffc00000 and
    # 0x7fc00001 come back as 0x7fc00000); every other value is bit-exact. Zarr keeps all bits.
    m.spectra[np.isnan(m.spectra)] = np.float32("nan")


@FILE_PROPERTY
@given(spec=synthetic_maps(), spectra_format=st.sampled_from(["zarr", "csv"]))
@example(
    spec={
        "spectra": np.zeros((4, 4), dtype=np.float32),
        "positions": np.zeros((4, 2), dtype=np.float32),
        "layout": "classic",
        "first_x": 4000.0,  # stored descending: the reader reverses it, and so must the unpacked folder
        "last_x": 650.0,  # (the background wavenumbers came back 2e-13 off, found by the audit)
        "titles": ["a", "b", "c", "d"],
        "z": None,
        "start_centiseconds": 8878,
    },
    spectra_format="zarr",
)
def test_written_map_survives_read_and_unpack_bit_exactly(tmp_path_factory, spec, spectra_format):
    folder = tmp_path_factory.mktemp("property")
    write_spec(folder / "m.map", spec)
    m = read_omnic(folder / "m.map")
    n = len(spec["spectra"])
    # oracle: the generator's own inputs (a descending file is reversed to ascending cm-1)
    stored = spec["spectra"][:, ::-1] if spec["first_x"] > spec["last_x"] else spec["spectra"]
    np.testing.assert_array_equal(m.spectra.view(np.uint32), stored.view(np.uint32))
    np.testing.assert_array_equal(m.x, spec["positions"][:, 0].astype(np.float64))
    np.testing.assert_array_equal(m.y, spec["positions"][:, 1].astype(np.float64))
    assert m.titles == spec["titles"]
    np.testing.assert_array_equal(m.times_s, 90.0 * np.arange(n))  # the writer steps the counter by 9000 per spectrum
    if spec["z"] is not None:
        np.testing.assert_array_equal(m.z, np.array(spec["z"], dtype=np.float32).astype(np.float64))
    # oracle: the axis runs between the two header values (float32 in the file, so 1 ulp = 6e-8 relative)
    expected_axis = np.linspace(min(spec["first_x"], spec["last_x"]), max(spec["first_x"], spec["last_x"]), len(m.axis))
    np.testing.assert_allclose(m.axis, expected_axis, rtol=1e-6)
    assert np.all(np.diff(m.axis) > 0)

    m.unpack(folder / "out", spectra=spectra_format)
    back = read_unpacked(folder / "out")
    if spectra_format == "csv":
        match_csv_precision(m, back)
    assert_same_map(m, back)


# ------------------------------------------------------------------ grid layout
ORDERS = ["raster", "serpentine", "column-major", "shuffled"]


@st.composite
def acquired_grids(draw):
    """(nx, ny, x, y, ideal column, ideal row): a jittered grid in one acquisition order, float32 positions."""
    nx, ny = draw(st.integers(2, 7)), draw(st.integers(2, 7))
    dx, dy = draw(st.floats(0.05, 200)), draw(st.floats(0.05, 200))
    x0, y0 = (float(np.float32(draw(st.floats(-3e4, 3e4)))) for _ in range(2))
    columns, rows = np.meshgrid(np.arange(nx), np.arange(ny))  # row-major: raster order
    columns, rows = columns.ravel(), rows.ravel()
    order = draw(st.sampled_from(ORDERS))
    if order == "serpentine":
        columns = np.where(rows % 2 == 1, nx - 1 - columns, columns)
    elif order == "column-major":
        columns, rows = np.meshgrid(np.arange(nx), np.arange(ny))
        columns, rows = columns.T.ravel(), rows.T.ravel()
    elif order == "shuffled":
        perm = np.array(draw(st.permutations(range(nx * ny))))
        columns, rows = columns[perm], rows[perm]
    # map_kinds.md: coordinates closer than 0.5 um and than a tenth of the largest gap are one row or column. Draw an
    # offset of at most +-jitter around the lattice, so two points of one column differ by at most 2 * jitter, and
    # only where the step leaves room (step >= 10 um).
    jitter = [
        draw(st.floats(0, 1, exclude_max=True)) * min(0.25, step / 20) if step >= 10 else 0.0 for step in (dx, dy)
    ]
    offsets = np.array(draw(st.lists(st.floats(-1, 1), min_size=2 * nx * ny, max_size=2 * nx * ny)))
    x = (x0 + dx * columns + jitter[0] * offsets[: nx * ny]).astype(np.float32).astype(np.float64)
    y = (y0 + dy * rows + jitter[1] * offsets[nx * ny :]).astype(np.float32).astype(np.float64)
    return nx, ny, x, y, columns, rows


@settings(max_examples=300)  # the cheap pure function: more draws than the file round trips
@given(acquired_grids())
def test_grid_layout_recovers_the_cell_of_every_point_in_any_acquisition_order(grid):
    nx, ny, x, y, columns, rows = grid
    layout = grid_layout(x, y)
    assert layout is not None, "a complete grid is not recognized"
    shape, (row, col) = layout
    assert shape == (ny, nx)
    # oracle: the lattice cell each point was generated for; rows and columns are numbered by ascending Y and X
    np.testing.assert_array_equal(row, rows)
    np.testing.assert_array_equal(col, columns)


# ------------------------------------------------------------------ CSV text and world files
formula_prefixes = st.sampled_from(["", "'", "=", "+", "-", "@", "\t", "\r", "'=", "''"])
any_text = st.text(st.characters(exclude_categories=["Cs"]), max_size=40)
csv_texts = st.builds(lambda prefix, text: prefix + text, formula_prefixes, any_text)


@given(csv_texts)
def test_csv_cell_is_undone_by_the_readers_unescape_and_never_starts_a_formula(text):
    cell = csv_cell(text)
    assert _unescape_cell(cell) == text
    # a spreadsheet runs a cell that starts with one of these, so it is quoted instead
    assert not cell.startswith(("=", "+", "-", "@", "\t", "\r"))


@given(csv_texts.filter(lambda text: "\0" not in text))
def test_csv_file_written_and_read_back_gives_the_escaped_cell(tmp_path_factory, text):
    path = tmp_path_factory.mktemp("cell") / "cell.csv"
    write_csv(path, ["title"], [[text]])
    with open(path, newline="", encoding="utf-8") as file:
        header, (cell,) = csv.reader(file)
    assert header == ["title"]
    assert _unescape_cell(cell) == text


@given(
    xmin=st.floats(-3e4, 3e4),
    width_um=st.floats(1e-3, 1e4),
    ymin=st.floats(-3e4, 3e4),
    height_um=st.floats(1e-3, 1e4),
    width=st.integers(1, 5000),
    height=st.integers(1, 5000),
)
def test_world_file_text_inverts_to_the_extent(xmin, width_um, ymin, height_um, width, height):
    extent = (xmin, xmin + width_um, ymin, ymin + height_um)
    pixel_x, _, _, minus_pixel_y, x_centre, y_centre = (
        float(v) for v in world_file_text(extent, width, height).split()
    )
    x_edge, y_edge = x_centre - pixel_x / 2, y_centre - minus_pixel_y / 2  # upper-left corner: (xmin, ymax)
    back = (x_edge, x_edge + pixel_x * width, y_edge + minus_pixel_y * height, y_edge)
    # 12 significant digits are written, so each number is off by at most 5e-12 relative to its own size. Rebuilding an
    # edge adds the centre (error 5e-12 |centre|), half a pixel and a whole pixel times the pixel count (5e-12 of the
    # size each, the count scales the error): at most 5e-12 (|centre| + 1.5 size), and size <= 2 max|coordinate|.
    scale = max(abs(v) for v in extent)
    np.testing.assert_allclose(back, extent, rtol=5e-12, atol=2e-11 * scale)
    assert (pixel_x, -minus_pixel_y) == pytest.approx((width_um / width, height_um / height), rel=5e-12)


# ------------------------------------------------------------------ Zarr
@FILE_PROPERTY
@given(synthetic_maps())
def test_zarr_store_opens_identical_to_to_xarray(tmp_path_factory, spec):
    folder = tmp_path_factory.mktemp("zarr")
    write_spec(folder / "m.map", spec)
    m = read_omnic(folder / "m.map")
    m.to_zarr(folder / "m.zarr")
    with xr.open_zarr(folder / "m.zarr") as back:
        xr.testing.assert_identical(back.load(), m.to_xarray())  # identical: values (NaN equal), dims, coords, attrs
