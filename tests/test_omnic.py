"""ftir_map.omnic.read_omnic on synthetic files (both container layouts) and on the real files in tests/data."""

import datetime as dt
import io
import json
import os
import re
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ftir_map import omnic
from ftir_map.omnic import OmnicFormatError, grid_layout, read_omnic

from .omnic_writer import LAYOUTS as FILE_LAYOUTS
from .omnic_writer import (
    _write,
    acquisition_block,
    build_blocks,
    map_display,
    point_array,
    spectral_header,
    tiny_bmp,
    write_map,
)

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
    np.testing.assert_array_equal(m.axis, np.linspace(650.0, 4000.0, N_WN))  # 650 and 4000 are exact in float32
    np.testing.assert_array_equal(m.axis, m.wavenumber)
    assert m.axis_unit == "cm-1"
    assert m.titles[0] == f"Spectrum 1 of {len(positions)}"
    np.testing.assert_allclose(m.times_s, 90.0 * np.arange(len(positions)))
    assert m.data_kind == "absorbance"
    assert m.collection.n_sample_scans == 16
    assert m.collection.mirror_velocity == float(np.float32(2.5317))  # stored as float32


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
    np.testing.assert_array_equal(bkg.spectrum, np.linspace(1.0, 2.0, N_WN, dtype=np.float32))  # stored as float32
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


def test_points_without_a_definition_that_are_not_collinear(tmp_path):
    path = tmp_path / "scattered.map"
    write_map(path, spectra(4), POINTS, kind=None)
    m = read_omnic(path)
    assert m.map_kind == "points" and m.grid_shape is None


@pytest.mark.parametrize("step", [0.4, 0.05, 1.0, 25.0])
def test_fine_grid_steps_are_still_a_grid(tmp_path, step):
    positions = [[1000.0 + step * i, 2000.0 + step * j] for j in range(3) for i in range(4)]
    path = tmp_path / "fine.map"
    write_map(path, spectra(12), positions, kind=None)
    m = read_omnic(path)
    assert m.map_kind == "grid" and m.grid_shape == (3, 4)


def test_jittered_grid_is_still_a_grid():
    rng = np.random.default_rng(0)
    # map_kinds.md merges coordinates only below 0.5 um, so jitter within +-0.25 is the safe range (+-0.3 fails on
    # 94 of 2000 seeds, when two jittered coordinates of one column are more than 0.5 um apart)
    x = np.array([25.0 * i + rng.uniform(-0.2, 0.2) for _ in range(3) for i in range(4)])
    y = np.array([25.0 * j + rng.uniform(-0.2, 0.2) for j in range(3) for _ in range(4)])
    assert grid_layout(x, y)[0] == (3, 4)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_descending_wavenumber_axis_is_reversed_with_spectra_and_backgrounds(tmp_path, layout):
    path = tmp_path / "descending.map"
    write_map(path, spectra(4), POINTS, layout=layout, first_x=4000.0, last_x=650.0)
    m = read_omnic(path)
    # linspace(4000, 650)[::-1] and linspace(650, 4000) differ by at most 2 float64 ulps (relative 2 * 2.2e-16)
    np.testing.assert_allclose(m.axis, np.linspace(650.0, 4000.0, N_WN), rtol=4.5e-16, atol=0)
    np.testing.assert_array_equal(m.spectra, spectra(4)[:, ::-1])
    (bkg,) = m.backgrounds
    np.testing.assert_array_equal(bkg.wavenumber, m.axis)  # the same linspace, reversed the same way
    stored = np.linspace(1.0, 2.0, N_WN, dtype=np.float32)  # the writer stores 1 -> 2 at 4000 -> 650
    np.testing.assert_array_equal(bkg.spectrum, stored[::-1])


def test_time_counter_wraps_at_2_to_the_32(tmp_path):
    path = tmp_path / "wrap.map"
    write_map(path, spectra(4), POINTS, start_centiseconds=2**32 - 9000)
    np.testing.assert_allclose(read_omnic(path).times_s, [0.0, 90.0, 180.0, 270.0])


def test_position_count_mismatch_warns_and_keeps_nan(tmp_path):
    path = tmp_path / "short_positions.map"
    write_map(path, spectra(4), POINTS[:3], kind=None)
    with pytest.warns(UserWarning, match=r"\b3 stage positions for 4 spectra") as record:
        m = read_omnic(path)
    assert "16-bit" not in str(record[0].message)  # that note is only for more than 65535 spectra
    assert np.isnan(m.x).all() and np.isnan(m.y).all() and m.y.dtype == np.float64 and m.map_kind == "points"
    assert m.background_index is not None  # the other arrays are fine


def test_background_index_count_mismatch_warns(tmp_path):
    path = tmp_path / "bkg.map"
    write_map(path, spectra(4), POINTS)
    data = bytearray(path.read_bytes())
    marker = struct.pack("<3H", 4, 1, 0)  # the index-3 array of 4 spectra
    at = data.index(marker)
    data[at : at + 2] = struct.pack("<H", 3)
    path.write_bytes(bytes(data))
    with pytest.warns(UserWarning, match=r"\b3 background indices for 4 spectra"):
        assert read_omnic(path).background_index is None


def test_more_than_65535_spectra_warn_about_the_16_bit_count(tmp_path):
    n = 65537
    path = tmp_path / "many.map"
    write_map(path, np.zeros((n, 2), dtype=np.float32), np.zeros((n, 2)), first_x=650.0, last_x=4000.0)
    with pytest.warns(UserWarning, match="16-bit"):
        m = read_omnic(path)
    assert m.n_points == n and np.isnan(m.x).all()


def hostile_files(tmp_path):
    """Malformed files that used to raise struct.error, IndexError or a NumPy error: name -> path."""
    out = {}

    def add(name, data):
        out[name] = tmp_path / f"{name}.map"
        out[name].write_bytes(data)

    add("magic_only", b"Spectral Exte File" + bytes(100))
    add("classic_magic_only", b"Spectral Data File")
    write_map(tmp_path / "ok.map", spectra(6), GRID, kind="area")
    data = bytearray((tmp_path / "ok.map").read_bytes())
    struct.pack_into("<H", data, 0x126, 60000)
    add("huge_directory", bytes(data[:5000]))
    out["short_header"] = tmp_path / "short_header.map"
    _write(out["short_header"], "classic", [(3, 1, bytes(148)), (2, 1, spectral_header(37, 650, 4000)[:40])])
    out["short_series"] = tmp_path / "short_series.map"
    _write(out["short_series"], "extended", [(2, 1, spectral_header(4, 650, 4000)), (0x12D, 1, bytes(100))])
    out["image_2_bytes"] = tmp_path / "image_2_bytes.map"
    blocks = [(2, 1, spectral_header(4, 650, 4000)), (0x94, 1, bytes(96)), (3, 1, bytes(16)), (0x93, 1, b"\x01\x00")]
    _write(out["image_2_bytes"], "classic", blocks)
    out["zero_points"] = tmp_path / "zero_points.map"
    blocks = [(2, 1, spectral_header(0, 650, 4000)), (0x94, 1, bytes(96)), (3, 1, bytes(0))]
    _write(out["zero_points"], "classic", blocks)
    return out


HOSTILE = [
    "magic_only",
    "classic_magic_only",
    "huge_directory",
    "short_header",
    "short_series",
    "image_2_bytes",
    "zero_points",
]


@pytest.mark.parametrize("name", HOSTILE)
def test_malformed_files_raise_omnic_format_error(tmp_path, name):
    with pytest.raises(OmnicFormatError):
        read_omnic(hostile_files(tmp_path)[name])


def test_struct_errors_from_parsing_become_omnic_format_errors(tmp_path):
    """The safety net: a malformed block no size check anticipated still gives OmnicFormatError."""
    path = tmp_path / "map.map"
    write_map(path, spectra(4), POINTS)

    def boom(*args, **kwargs):
        raise struct.error("unpack requires a buffer of 4 bytes")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(omnic, "_map_definition", boom)
        with pytest.raises(OmnicFormatError, match="malformed file"):
            read_omnic(path)


def test_grid_layout_rejects_incomplete_grids():
    x, y = np.array([0.0, 25.0, 50.0, 0.0, 25.0]), np.array([0.0, 0.0, 0.0, 50.0, 50.0])
    assert grid_layout(x, y) is None
    assert grid_layout(np.append(x, 50.0), np.append(y, 50.0))[0] == (2, 3)


# ------------------------------------------------------------------ decoded fields (file_format.md sections 3, 4, 5)
def read_blocks(tmp_path, blocks, layout="extended"):
    path = tmp_path / "blocks.map"
    _write(path, layout, blocks)
    return read_omnic(path)


def with_block(blocks, key, index, content):
    """The block list with the (key, index) block replaced."""
    assert any((k, i) == (key, index) for k, i, _ in blocks)
    return [(k, i, content if (k, i) == (key, index) else c) for k, i, c in blocks]


def without_block(blocks, key, index):
    assert any((k, i) == (key, index) for k, i, _ in blocks)
    return [(k, i, c) for k, i, c in blocks if (k, i) != (key, index)]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_collection_parameter_is_decoded_from_its_own_offset(tmp_path, layout):
    """Each field of the writer's header has its own value, so two fields read from the wrong offsets differ."""
    path = tmp_path / "map.map"
    write_map(path, spectra(4), POINTS, layout=layout)
    c = read_omnic(path).collection
    # the values the writer puts at the offsets of file_format.md section 3 (floats are stored as float32)
    assert (c.n_points, c.x_units, c.data_kind) == (N_WN, "cm-1", "absorbance")  # +4, +8, +12
    assert (c.last_x, c.first_x) == (4000.0, 650.0)  # +16, +20 (exact in float32)
    assert c.n_scan_points == 8480  # +28
    assert c.interferogram_peak_position == 4096  # +32
    assert c.n_sample_scans == 16  # +36
    assert c.n_fft_points == 8192  # +44
    assert c.n_background_scans == 24  # +52
    assert c.background_gain == 1.5  # +56
    assert c.spectrum_set_identifier == 0.625  # +72
    assert c.laser_frequency == float(np.float32(15798.7))  # +80
    assert c.mirror_velocity == float(np.float32(2.5317))  # key 0x6a +48
    assert c.data_spacing == 3350.0 / 36  # (4000 - 650) / (37 - 1)
    assert set(c.raw) == {"spectral_header", "acquisition"}
    assert c.raw["spectral_header"] == spectral_header(N_WN, 650.0, 4000.0) and len(c.raw["spectral_header"]) == 140
    assert c.raw["acquisition"] == acquisition_block()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_unsigned_header_fields_above_2_to_the_31_are_not_negative(tmp_path, layout):
    fields = {
        "n_scans": 2**31 + 7,
        "n_scan_points": 2**32 - 1,
        "peak_position": 2**31,
        "background_scans": 4_000_000_000,
    }
    path = tmp_path / "map.map"
    write_map(path, spectra(4), POINTS, layout=layout, header_fields=fields | {"fft_points": 3_000_000_000})
    c = read_omnic(path).collection
    assert (c.n_sample_scans, c.n_scan_points) == (2**31 + 7, 2**32 - 1)
    assert (c.interferogram_peak_position, c.n_background_scans, c.n_fft_points) == (
        2**31,
        4_000_000_000,
        3_000_000_000,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_unknown_unit_codes_name_their_own_code(tmp_path, layout):
    path = tmp_path / "map.map"
    write_map(path, spectra(4), POINTS, layout=layout, header_fields={"x_units_code": 7, "y_units_code": 99})
    m = read_omnic(path)
    assert m.collection.x_units == "unknown (code 7)" and m.axis_unit == "unknown (code 7)"
    assert m.collection.data_kind == m.data_kind == "unknown (code 99)"


@pytest.mark.parametrize("size", [51, 52])
def test_acquisition_block_needs_52_bytes_for_the_velocity(tmp_path, size):
    blocks = with_block(build_blocks(spectra(4), POINTS), 0x6A, 1, acquisition_block(size=size))
    velocity = read_blocks(tmp_path, blocks).collection.mirror_velocity
    assert velocity == (float(np.float32(2.5317)) if size == 52 else None)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_map_definition_field_is_decoded_from_its_own_offset(tmp_path, layout):
    path = tmp_path / "grid.map"
    write_map(path, spectra(6), GRID, layout=layout, kind="area")
    d = read_omnic(path).map_definition
    assert d == omnic.MapDefinition(  # the values the writer puts at the offsets of file_format.md section 5
        kind="area",  # +6
        x_range=(100.0, 150.0),  # +8: first and last position of GRID
        y_range=(200.0, 250.0),
        step=25.0,  # +88
        x_step=50.0,  # +112
        y_step=75.0,  # +120
        n_x=7,  # +128
        n_y=9,  # +132
        title="synthetic map",  # +180
        file_name="synthetic.map",  # +586
        directory="C:\\synthetic\\",  # +844
    )


def test_unknown_map_kind_code_above_2_to_the_15_is_named_by_its_code(tmp_path):
    path = tmp_path / "unknown_kind.map"
    write_map(path, spectra(4), POINTS, kind=40000)
    definition = read_omnic(path).map_definition
    assert definition is not None and definition.kind == "unknown (40000)"


def test_map_definition_counts_above_2_to_the_31_are_not_negative(tmp_path):
    path = tmp_path / "huge_counts.map"
    write_map(path, spectra(6), GRID, kind="area", definition_fields={"n_x": 2**31 + 1, "n_y": 2**32 - 1})
    d = read_omnic(path).map_definition
    assert d is not None and (d.n_x, d.n_y) == (2**31 + 1, 2**32 - 1)


def test_text_fields_of_the_map_definition_stop_at_256_bytes(tmp_path):
    """A full 256-byte title, file name and directory must not pick up the byte after it (file_format.md section 5)."""
    fields = {"title": "t" * 256, "file_name": "f" * 256, "directory": "d" * 256}
    blocks = build_blocks(spectra(6), GRID, kind="area", definition_fields=fields)
    definition = bytearray({(k, i): c for k, i, c in blocks}[(0x82, 0)])
    for end in (180 + 256, 586 + 256, 844 + 256):
        definition[end] = ord("X")
    d = read_blocks(tmp_path, with_block(blocks, 0x82, 0, bytes(definition))).map_definition
    assert (d.title, d.file_name, d.directory) == ("t" * 256, "f" * 256, "d" * 256)


@pytest.mark.parametrize(("size", "expected"), [(1099, None), (1100, "area")])
def test_map_definition_needs_1100_bytes(tmp_path, size, expected):
    blocks = build_blocks(spectra(6), GRID, kind="area", definition_fields={"size": size})
    definition = read_blocks(tmp_path, blocks).map_definition
    assert (definition.kind if definition else None) == expected


def test_map_description_is_decoded_in_the_order_y0_y1_dy_x0_x1_dx(tmp_path):
    """Block 0x145 holds six float32 (y0, y1, dy, x0, x1, dx) at +66 (file_format.md section 2)."""
    description = ((10.0, 20.0), (110.0, 220.0), 5.0, 7.0)  # (x0, y0), (x1, y1), dx, dy
    blocks = build_blocks(spectra(4), POINTS, description=description)
    assert read_blocks(tmp_path, blocks).map_description == omnic.MapDescription(*description)


@pytest.mark.parametrize(
    ("size", "labels", "decoded"),
    [
        (240, (b"Position", b"Position"), True),
        (239, (b"Position", b"Position"), False),
        (264, (b"Position", b"Distance"), False),
        (264, (b"Distance", b"Position"), False),
    ],
    ids=["240 bytes", "239 bytes", "second label wrong", "first label wrong"],
)
def test_map_description_needs_240_bytes_and_both_position_labels(tmp_path, size, labels, decoded):
    block = map_display((10.0, 20.0), (110.0, 220.0), 5.0, 7.0, labels=labels, size=size)
    blocks = [*build_blocks(spectra(4), POINTS), (0x145, 1, block)]
    assert (read_blocks(tmp_path, blocks).map_description is not None) == decoded


def test_a_definition_that_says_points_wins_over_a_complete_grid(tmp_path):
    path = tmp_path / "points_on_a_grid.map"
    write_map(path, spectra(6), GRID, kind="points")
    m = read_omnic(path)
    assert m.map_kind == "points" and m.grid_shape is None and m.grid_index is None


# ------------------------------------------------------------------ size bounds (file_format.md, "malformed blocks")
def test_header_of_exactly_304_bytes_is_read_and_303_is_too_short():
    magic = b"Spectral Data File"
    assert omnic.read_directory(magic + bytes(304 - 18)) == ("classic", [])  # 0 entries: nothing to read
    with pytest.raises(OmnicFormatError, match="303 bytes is shorter than its 304-byte header"):
        omnic.read_directory(magic + bytes(303 - 18))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_directory_that_ends_exactly_at_the_end_of_the_file_is_read(layout):
    magic, fmt = FILE_LAYOUTS[layout]
    data = bytearray(0x130 + 2 * struct.calcsize(fmt))
    data[:18] = magic
    struct.pack_into("<H", data, 0x126, 2)
    for i in range(2):  # two empty blocks whose offset is the end of the file
        struct.pack_into(fmt, data, 0x130 + i * struct.calcsize(fmt), 0x71, len(data), 0, i, 0)
    assert len(omnic.read_directory(bytes(data))[1]) == 2
    with pytest.raises(OmnicFormatError, match="block directory"):
        omnic.read_directory(bytes(data[:-1]))


def test_directory_count_above_2_to_the_15_is_not_negative(tmp_path):
    blocks = [*build_blocks(spectra(4), POINTS), *[(0x71, 0, b"")] * 40000]
    m = read_blocks(tmp_path, blocks)
    assert len(m.blocks) == len(blocks) and m.n_points == 4


def test_a_directory_count_larger_than_the_file_is_reported_as_the_directory(tmp_path):
    with pytest.raises(OmnicFormatError, match="block directory"):
        read_omnic(hostile_files(tmp_path)["huge_directory"])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_file_cut_by_one_byte_raises(tmp_path, layout):
    path = tmp_path / "map.map"
    write_map(path, spectra(4), POINTS, layout=layout)
    assert read_omnic(path).n_points == 4
    path.write_bytes(path.read_bytes()[:-1])  # the last block now ends one byte after the end of the file
    with pytest.raises(OmnicFormatError, match="points past the end of the file"):
        read_omnic(path)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_spectral_header_that_ends_exactly_at_the_end_of_the_file_is_read(tmp_path, layout):
    blocks = build_blocks(spectra(4), POINTS, layout=layout)
    header = {(k, i): c for k, i, c in blocks}[(2, 1)]
    blocks = [*without_block(blocks, 2, 1), (2, 1, header)]  # the header is now the last block in the file
    assert read_blocks(tmp_path, blocks, layout).n_points == 4


def test_extended_spectra_block_is_a_196_byte_prefix_and_whole_records(tmp_path):
    header = spectral_header(N_WN, 650.0, 4000.0)
    record = 96 + 4 + 4 * N_WN
    prefix = header + bytes(56)
    assert len(prefix) == 196
    with pytest.raises(OmnicFormatError, match="is not 196 \\+ n x"):  # no record at all
        read_blocks(tmp_path, [(2, 1, header), (0x12D, 1, prefix)])
    with pytest.raises(OmnicFormatError, match="is not 196 \\+ n x"):  # one byte too many
        read_blocks(tmp_path, [(2, 1, header), (0x12D, 1, prefix + bytes(record) + b"\0")])
    m = read_blocks(tmp_path, [(2, 1, header), (0x12D, 1, prefix + bytes(record))])  # exactly one record
    assert m.n_points == 1 and m.spectra.shape == (1, N_WN)


def test_classic_spectrum_without_a_header_block_has_an_empty_title(tmp_path):
    blocks = without_block(build_blocks(spectra(3), POINTS[:3], layout="classic"), 0x94, 2)
    m = read_blocks(tmp_path, blocks, "classic")
    assert m.titles == ["Spectrum 1 of 3", "", "Spectrum 3 of 3"]
    np.testing.assert_array_equal(m.spectra, spectra(3))


# ------------------------------------------------------------------ key-6 arrays
@pytest.mark.parametrize(
    "bad_block",
    [b"\0" * 5, struct.pack("<3H", 1000, 2, 0), struct.pack("<3H", 3, 0, 0)],
    ids=["shorter than its 6-byte header", "count larger than the block", "0 values per point"],
)
def test_a_bad_key_6_block_is_skipped_not_the_blocks_after_it(tmp_path, bad_block):
    blocks = [(6, 0, bad_block), *build_blocks(spectra(4), POINTS)]
    np.testing.assert_array_equal(read_blocks(tmp_path, blocks).x, as_stored(POINTS)[:, 0])


def test_an_empty_key_6_array_of_exactly_6_bytes_counts_as_zero_positions(tmp_path):
    blocks = with_block(build_blocks(spectra(4), POINTS), 6, 1, struct.pack("<3H", 0, 2, 0))
    with pytest.warns(UserWarning, match=r"\b0 stage positions for 4 spectra"):
        assert np.isnan(read_blocks(tmp_path, blocks).x).all()


def test_a_key_6_array_that_fits_its_block_exactly_is_read_and_one_byte_less_is_not(tmp_path):
    exact = point_array(POINTS, 2)[:-2]  # the writer adds 2 trailing bytes; without them the array fills the block
    assert len(exact) == 6 + 4 * 2 * len(POINTS)
    blocks = build_blocks(spectra(4), POINTS)
    np.testing.assert_array_equal(read_blocks(tmp_path, with_block(blocks, 6, 1, exact)).x, as_stored(POINTS)[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a block too small for its count is skipped silently, not read past its end
        assert np.isnan(read_blocks(tmp_path, with_block(blocks, 6, 1, exact[:-1])).x).all()


def test_a_key_6_count_above_2_to_the_15_is_not_negative(tmp_path):
    n = 40000
    positions = np.column_stack([np.arange(n), np.arange(n) * 2.0])
    path = tmp_path / "many.map"
    write_map(path, np.zeros((n, 2), dtype=np.float32), positions)
    m = read_omnic(path)
    np.testing.assert_array_equal(m.x, positions[:, 0])
    np.testing.assert_array_equal(m.y, positions[:, 1])


def test_z_count_mismatch_warns_about_stage_z_values(tmp_path):
    path = tmp_path / "z.map"
    write_map(path, spectra(4), POINTS, layout="classic", z=[1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match=r"\b3 stage Z values for 4 spectra"):
        assert read_omnic(path).z is None


@pytest.mark.parametrize(("n", "stored", "note"), [(65535, 3, False), (65536, 0, True)])
def test_the_16_bit_note_starts_above_65535_spectra(tmp_path, n, stored, note):
    """65535 spectra still fit the 16-bit count (so this one is tampered to 3); 65536 wraps to 0 on its own."""
    positions = np.zeros((n, 2))
    blocks = build_blocks(np.zeros((n, 2), dtype=np.float32), positions)
    xy = bytearray(point_array(positions, 2))
    struct.pack_into("<H", xy, 0, stored)
    with pytest.warns(UserWarning) as record:  # the background indices (key 6, index 3) wrap too: 2 warnings
        read_blocks(tmp_path, with_block(blocks, 6, 1, bytes(xy)))
    (message,) = (str(w.message) for w in record if "stage positions" in str(w.message))
    assert re.match(rf"the file stores {stored} stage positions for {n} spectra\b", message)
    assert ("16-bit" in message) == note


# ------------------------------------------------------------------ backgrounds
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_background_spectrum_without_its_header_is_not_a_background(tmp_path, layout):
    blocks = without_block(build_blocks(spectra(4), POINTS, layout=layout), 2, 0)
    assert read_blocks(tmp_path, blocks, layout).backgrounds == []


def test_a_background_header_shorter_than_140_bytes_is_not_used(tmp_path):
    blocks = with_block(build_blocks(spectra(4), POINTS), 2, 0, spectral_header(N_WN, 650.0, 4000.0)[:100])
    assert read_blocks(tmp_path, blocks).backgrounds == []


def title_block(title, size, seconds):
    block = bytearray(size)
    block[: len(title)] = title
    if size >= 260:
        struct.pack_into("<I", block, 256, seconds)
    return bytes(block)


@pytest.mark.parametrize(("size", "has_title"), [(259, False), (260, True), (264, True)])
def test_background_title_block_needs_260_bytes(tmp_path, size, has_title):
    blocks = with_block(build_blocks(spectra(4), POINTS), 0x6B, 0, title_block(b"title", size, 86400))
    (background,) = read_blocks(tmp_path, blocks).backgrounds
    expected = ("title", dt.datetime(1900, 1, 1)) if has_title else ("", None)  # 86400 s after 1899-12-31
    assert (background.title, background.time) == expected


def test_a_256_byte_background_title_does_not_pick_up_the_time_bytes(tmp_path):
    seconds = 0xDC220E81  # its first byte (0x81) is not a null, so a 257-byte read would add a character
    blocks = with_block(build_blocks(spectra(4), POINTS), 0x6B, 0, title_block(b"T" * 256, 264, seconds))
    (background,) = read_blocks(tmp_path, blocks).backgrounds
    assert background.title == "T" * 256
    assert background.time == dt.datetime(1899, 12, 31) + dt.timedelta(seconds=seconds)


# ------------------------------------------------------------------ images
def test_image_block_of_exactly_4_bytes_is_an_empty_image(tmp_path):
    blocks = with_block(build_blocks(spectra(4), POINTS, images=[[0.0, 0.0, 4.0, 3.0]]), 0x93, 1, struct.pack("<I", 0))
    (image,) = read_blocks(tmp_path, blocks).images
    assert (image.data, image.width, image.height) == (b"", None, None)


def test_image_bytes_are_exactly_the_embedded_file(tmp_path):
    bmp = tiny_bmp(5, 2)
    path = tmp_path / "image.map"
    write_map(path, spectra(4), POINTS, images=[[0.0, 0.0, 5.0, 2.0]], image_bmp=bmp)
    (image,) = read_omnic(path).images
    assert image.data == bmp and (image.width, image.height) == (5, 2)


@pytest.mark.parametrize("length", [10**6, 2**31 + 5], ids=["longer than the block", "above 2**31"])
def test_image_length_field_larger_than_the_block_is_limited_to_the_block(tmp_path, length):
    bmp = tiny_bmp()
    blocks = build_blocks(spectra(4), POINTS, images=[[0.0, 0.0, 4.0, 3.0]], image_bmp=bmp)
    blocks = with_block(blocks, 0x93, 1, struct.pack("<I", length) + bmp)
    (image,) = read_blocks(tmp_path, blocks).images
    assert image.data == bmp


def test_images_without_a_rectangle_have_no_extent(tmp_path):
    rect = [0.0, 0.0, 4.0, 3.0]
    blocks = build_blocks(spectra(4), POINTS, images=[rect, rect])
    (first, second) = read_blocks(tmp_path, without_block(blocks, 6, 2)).images
    assert first.extent is None and second.extent is None and first.pixel_size_um is None
    one_rect = with_block(blocks, 6, 2, point_array([rect], 4))  # 2 images, 1 rectangle
    (first, second) = read_blocks(tmp_path, one_rect).images
    assert first.extent == tuple(rect) and second.extent is None


def encoded(size, fmt, **options):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, fmt, **options)
    return buffer.getvalue()


def pillow_size(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.size


@pytest.mark.parametrize("size", [(7, 3), (3, 7), (40000, 1)])
def test_bmp_size_equals_pillows(size):
    data = encoded(size, "BMP")
    assert omnic._image_size("BMP", data) == pillow_size(data) == size


def test_top_down_bmp_height_is_positive():
    data = bytearray(encoded((7, 3), "BMP"))
    struct.pack_into("<i", data, 22, -3)  # a negative height means rows stored top-down
    assert omnic._image_size("BMP", bytes(data)) == pillow_size(bytes(data)) == (7, 3)


def test_bmp_size_needs_26_bytes():
    data = encoded((7, 3), "BMP")
    assert omnic._image_size("BMP", data[:26]) == (7, 3)
    assert omnic._image_size("BMP", data[:25]) == (None, None)


@pytest.mark.parametrize("size", [(7, 3), (3, 7), (40000, 1)])
@pytest.mark.parametrize("options", [{}, {"progressive": True}], ids=["baseline", "progressive"])
def test_jpeg_size_equals_pillows(size, options):
    data = encoded(size, "JPEG", **options)
    assert omnic._image_size("JPEG", data) == pillow_size(data) == size


def test_jpeg_size_skips_segments_longer_than_32767_bytes():
    data = encoded((7, 3), "JPEG", icc_profile=b"p" * 40000)
    assert b"ICC_PROFILE" in data[:40] and data.index(b"\xff\xc0") > 40000  # the long segment precedes the frame
    assert omnic._image_size("JPEG", data) == pillow_size(data) == (7, 3)


@pytest.mark.parametrize("marker", [0xC1, 0xC2, 0xC3])
def test_jpeg_size_reads_every_start_of_frame_marker(marker):
    data = bytearray(encoded((7, 3), "JPEG"))
    data[data.index(b"\xff\xc0") + 1] = marker  # SOF0 -> SOF1, SOF2 or SOF3: same layout, other coding process
    assert omnic._image_size("JPEG", bytes(data)) == (7, 3)


def test_jpeg_size_of_truncated_files():
    frame = b"\xff\xd8\xff\xc0" + struct.pack(">HBHH", 8, 8, 3, 7)  # start of image, then the frame header (11 bytes)
    assert omnic._image_size("JPEG", frame + b"\0") == (7, 3)  # 12 bytes: the dimensions are complete
    assert omnic._image_size("JPEG", b"\xff\xd8") == (None, None)  # nothing after the start-of-image marker
    assert omnic._image_size("PNG", b"\x89PNG" + bytes(40)) == (None, None)


# ------------------------------------------------------------------ geometry thresholds (map_kinds.md)
def test_a_2_by_2_grid_is_a_grid():
    assert grid_layout(np.array([0.0, 25.0, 0.0, 25.0]), np.array([0.0, 0.0, 50.0, 50.0]))[0] == (2, 2)


@pytest.mark.parametrize(
    ("x", "y", "shape"),
    [
        ([0, 25, 50, 75], [0, 0, 0, 0], None),  # 1 row of 4: a line, not a grid
        ([0, 0, 0, 0], [0, 25, 50, 75], None),  # 4 rows of 1
        ([0, 25, 0, 25, 0, 25], [0, 0, 50, 50, 100, 100], (3, 2)),  # 3 rows of 2
        ([0, 25, 50, 0, 25, 50], [0, 0, 0, 50, 50, 50], (2, 3)),  # 2 rows of 3
    ],
    ids=["1 x 4", "4 x 1", "3 x 2", "2 x 3"],
)
def test_a_grid_needs_two_rows_and_two_columns(x, y, shape):
    layout = grid_layout(np.array(x, dtype=float), np.array(y, dtype=float))
    assert (layout[0] if layout else None) == shape


def test_coordinates_exactly_at_the_tolerance_are_one_column():
    """Two x values 0.5 um apart (the tolerance) belong to one column; the gap to the next column is 100 um."""
    layout = grid_layout(np.array([0.0, 100.5, 0.5, 100.0]), np.array([0.0, 0.0, 50.0, 50.0]))
    assert layout is not None and layout[0] == (2, 2)


def test_positions_that_are_not_finite_are_neither_a_grid_nor_a_line():
    x, y = np.array([0.0, 25.0, 0.0, 25.0]), np.array([0.0, 0.0, 50.0, np.nan])
    assert grid_layout(x, y) is None and grid_layout(y, x) is None
    assert not omnic.is_collinear(x, y) and not omnic.is_collinear(y, x)


@pytest.mark.parametrize(
    ("x", "y", "tol", "expected"),
    [
        ([0.0, 2.0], [0.0, 5.0], 0.5, True),  # 2 points far apart are a line
        ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0], 0.5, False),  # all at one place
        ([0.0, 0.5], [0.0, 0.0], 0.5, False),  # extent exactly the tolerance: still one place
        ([0.0, 0.2, 0.4], [0.0, 0.2, 0.4], 0.5, False),  # a short diagonal: each axis spans less than the tolerance
        ([0.0, 0.1, 0.5], [0.4, 0.4, 0.0], 0.5, False),  # largest axis span exactly the tolerance (0.4 and 0.5)
        ([0.0, 2.0, 4.0], [0.0, 0.0, 0.0], 0.0, True),  # on the line: distance 0 <= tolerance 0
        ([0.0, 100.0, 200.0], [0.0, 0.3, 0.0], 0.5, True),  # 0.2 um off the line: within the tolerance
        ([0.0, 100.0, 200.0], [0.0, 1.0, 0.0], 0.5, False),  # 0.67 um off the line
        ([0.0, 100.0, 200.0, 300.0], [0.0, 100.0, 200.0, 300.0], 0.5, True),  # a long diagonal
    ],
)
def test_is_collinear_thresholds(x, y, tol, expected):
    assert omnic.is_collinear(np.array(x), np.array(y), tol) is expected


# ------------------------------------------------------------------ real files (tests/data, see tests/data/README.md)
def real(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not in tests/data")
    return path


POINT_MAP = "GCB_1C_GC52_CPX best points-50x50um_256avg.map"
GRID_MAP = "GCB_1C_ol54_ultrafast map_75um aperture and step.map"
LINE_MAP = "GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map"
IFG_MAP = "test file GCB_1C_GC52_opx profile1_50x50um_25um step (overlapping points)_256 averages_Ifg.map"
CSV_DIR = DATA / "cpx52_best_points_csv"
FLOAT32_HALF_ULP = 2.0**-24  # relative rounding error of a value stored as float32


def title_time(title):
    """A date title such as "Wed Jan 11 16:34:11 2017 (GMT+00:00)" as a datetime (resolution 1 s)."""
    return dt.datetime.strptime(title.replace(" (GMT+00:00)", ""), "%a %b %d %H:%M:%S %Y")


def test_real_point_map_matches_omnic_csv_export():
    m = read_omnic(real(POINT_MAP))
    assert (m.layout, m.map_kind, m.spectra.shape) == ("classic", "points", (6, 1725))
    csvs = sorted(CSV_DIR.glob("*avg000[1-6].CSV"))
    assert len(csvs) == 6
    for i, csv in enumerate(csvs):
        exported = np.loadtxt(csv, delimiter=",")
        # OMNIC writes 7 significant digits ("6.749918e+002"): a relative rounding of at most 5e-7, plus the float32
        # rounding of the stored axis (6e-8). Measured: 5.3e-7 on the axis.
        np.testing.assert_allclose(m.axis, exported[:, 0], rtol=5.6e-7, atol=0)
        # the same 7 digits on the spectra; measured 4.95e-7
        np.testing.assert_allclose(m.spectra[i], exported[:, 1], rtol=5e-7, atol=0)
    bkg = np.loadtxt(CSV_DIR / "GCB_1C_GC52_CPX best points-50x50um_256avgbkg1.CSV", delimiter=",")
    np.testing.assert_allclose(m.backgrounds[0].spectrum, bkg[:, 1], rtol=5e-7, atol=0)  # measured 2.4e-7
    # The six X values shown in OMNIC (2 decimals: rounding error at most 0.005 um; 429.375 shows as 429.38)
    np.testing.assert_allclose(m.x, [556.36, 281.99, 291.36, 429.38, 481.05, 509.80], rtol=0, atol=0.005)
    assert m.z is not None
    # Z values as shown by OMNIC (exactly representable in float32); the next test checks them against block 0x82
    np.testing.assert_allclose(m.z, [20507.25, 20440.5, 20193.75, 20496.0, 20484.75, 20272.5])
    # pixel size to the 4 decimals measured on this file (golden value); file_format.md section 8 gives 0.581 x 0.589
    assert len(m.images) == 2 and m.images[0].pixel_size_um == pytest.approx((0.5809, 0.5889), abs=1e-4)
    assert m.images[0].pixel_size_um == pytest.approx((0.581, 0.589), abs=5e-4)  # the documented 3 decimals


def test_real_point_map_positions_and_z_match_the_point_list_in_the_map_definition():
    """Block 0x82 holds the point list as float64: count at +1154, X at +1158, Y 6 values later, Z at +1414."""
    m = read_omnic(real(POINT_MAP))
    (block,) = m.block_bytes(0x82)
    assert struct.unpack_from("<I", block, 1154) == (6,)
    x, y = np.frombuffer(block, "<f8", 6, 1158), np.frombuffer(block, "<f8", 6, 1158 + 48)
    # the stage positions are these float64 values rounded to float32 (relative error at most 2**-24)
    np.testing.assert_allclose(m.x, x, rtol=FLOAT32_HALF_ULP, atol=0)
    np.testing.assert_allclose(m.y, y, rtol=FLOAT32_HALF_ULP, atol=0)
    assert m.z is not None
    np.testing.assert_array_equal(m.z, np.frombuffer(block, "<f8", 6, 1414))  # also stored as float64: equal


def test_real_point_map_title_name_and_scans_agree_with_the_file_name():
    m = read_omnic(real(POINT_MAP))
    assert m.title == m.path.stem  # the title at 0x1E is the file name without ".map"
    assert m.map_definition is not None and m.map_definition.file_name == m.path.name
    assert m.map_definition.title == m.path.stem
    stated = re.search(r"(\d+)avg", m.path.name)  # "256avg"
    assert stated is not None and m.collection.n_sample_scans == int(stated.group(1))


def test_real_grid_map():
    m = read_omnic(real(GRID_MAP))
    assert (m.layout, m.map_kind, m.grid_shape, m.spectra.shape) == ("extended", "grid", (13, 11), (143, 432))
    assert m.map_definition is not None
    assert m.map_definition.kind == "area" and (m.map_definition.n_x, m.map_definition.n_y) == (11, 13)
    assert len(m.images) == 6 and {im.format for im in m.images} == {"JPEG"}


def test_real_grid_map_agrees_with_its_own_definition_and_file_name():
    m = read_omnic(real(GRID_MAP))
    definition, description = m.map_definition, m.map_description
    assert definition is not None and description is not None
    assert definition.file_name == m.path.name and definition.title == m.path.stem
    assert definition.x_step == definition.y_step == 75.0  # "75um aperture and step" in the file name
    # block 0x145 repeats the first position and the steps (all float32 values, so equal)
    assert description.first_location == (m.x[0], m.y[0])
    assert (description.x_step, description.y_step) == (definition.x_step, definition.y_step)
    ds = m.to_xarray()
    # a difference of two float32-rounded positions is off by at most one float32 spacing at the larger value
    np.testing.assert_allclose(np.diff(ds["x"].values), definition.x_step, rtol=0, atol=np.spacing(np.float32(1874.3)))
    np.testing.assert_allclose(np.diff(ds["y"].values), definition.y_step, rtol=0, atol=np.spacing(np.float32(20400.2)))
    # the first column and row centers are stored as float64 in the definition: the grid axes are those, to float32
    x0, y0 = definition.x_range[0], definition.y_range[0]
    assert abs(ds["x"].values[0] - x0) <= FLOAT32_HALF_ULP * x0  # measured 6.0e-5 um
    assert abs(ds["y"].values[0] - y0) <= FLOAT32_HALF_ULP * y0  # measured 8.0e-4 um


def test_real_line_scan():
    m = read_omnic(real(LINE_MAP))
    assert (m.map_kind, m.spectra.shape) == ("line", (21, 1725))
    assert m.map_definition is not None
    assert m.map_definition.kind == "line" and m.map_definition.step == 50.0
    assert m.map_definition.file_name == m.path.name and m.map_definition.title == m.path.stem
    # The stored positions step 49.9964 um (the start and end in the definition are 999.93 um apart for 20 steps),
    # not exactly the nominal 50; each difference of two float32 positions near 2.4e4 um is off by < 0.00196 um.
    (x0, x1), (y0, y1) = m.map_definition.x_range, m.map_definition.y_range
    step = np.hypot(x1 - x0, y1 - y0) / (m.n_points - 1)
    np.testing.assert_allclose(np.hypot(np.diff(m.x), np.diff(m.y)), step, rtol=0, atol=0.002)


def test_real_background_times_equal_their_date_titles():
    for name in (GRID_MAP, LINE_MAP):
        (background,) = read_omnic(real(name)).backgrounds
        assert background.time == title_time(background.title)  # the titles have 1 s resolution: exact, not close


def test_real_interferogram_map():
    m = read_omnic(real(IFG_MAP))
    assert (m.data_kind, m.map_kind, m.spectra.shape) == ("interferogram", "line", (12, 8192))
    assert m.titles[1] == "Wed Jan 11 16:34:11 2017 (GMT+00:00)"
    # the time counter against the date titles: each title is truncated to 1 s, so a difference is off by < 1 s
    seconds = [(title_time(t) - title_time(m.titles[0])).total_seconds() for t in m.titles]
    np.testing.assert_allclose(m.times_s, seconds, rtol=0, atol=1.0)


def test_real_array_detector_map():
    m = read_omnic(real(ARRAY_MAP.name))
    assert (m.map_kind, m.grid_shape, m.spectra.shape) == ("array", (43, 31), (1333, 853))
    assert m.data_kind == "absorbance" and m.history.startswith("Converted to absorbance")
    title_xy = np.array([[float(t.split("X=")[1].split(",")[0]), float(t.split("Y=")[1])] for t in m.titles])
    # the titles carry 2 decimals (0.005 um) and the stored float32 positions near 2e3 um round by up to 1.2e-4 um
    np.testing.assert_allclose(np.column_stack([m.x, m.y]), title_xy, rtol=0, atol=0.0052)  # measured 0.0026
    assert m.background_index is not None
    assert m.background_index.max() == len(m.backgrounds) - 1 == 15
    # file_format.md section 8 gives 0.544 x 0.539 um per pixel (3 decimals, so 5e-4); measured 0.5443 x 0.5393
    for image in m.images:
        assert image.pixel_size_um == pytest.approx((0.544, 0.539), abs=5e-4)


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
