"""Write small synthetic OMNIC .map files with the block layout described in docs/site/file_format.md (test helper)."""

from __future__ import annotations

import datetime as dt
import struct

import numpy as np

EPOCH = dt.datetime(1899, 12, 31)
LAYOUTS = {"classic": (b"Spectral Data File", "<HIIIH"), "extended": (b"Spectral Exte File", "<HQIII")}
Y_UNITS = {"absorbance": 17, "single_beam": 15, "interferogram": 22}
MAP_KIND_CODES = {"points": 1, "line": 2, "area": 3}


def spectral_header(
    n_points,
    first_x,
    last_x,
    data_kind="absorbance",
    n_scans=16,
    *,
    background_scans=24,
    n_scan_points=8480,
    peak_position=4096,
    fft_points=8192,
    gain=1.5,
    laser=15798.7,
    identifier=0.625,
    x_units_code=None,
    y_units_code=None,
):
    """The 140-byte key-2 block (docs/site/file_format.md, section 3). Every field has its own default value, so a
    reader that mixes up two fields (sample scans +36 and background scans +52, say) gives a wrong number."""
    h = bytearray(140)
    x_units = x_units_code or (2 if data_kind == "interferogram" else 1)
    struct.pack_into(
        "<4I", h, 0, 1 if data_kind == "interferogram" else 3, n_points, x_units, y_units_code or Y_UNITS[data_kind]
    )
    struct.pack_into("<ff", h, 16, last_x, first_x)
    struct.pack_into("<3I", h, 28, n_scan_points, peak_position, n_scans)
    struct.pack_into("<I", h, 44, fft_points)
    struct.pack_into("<If", h, 52, background_scans, gain)
    struct.pack_into("<f", h, 72, identifier)
    struct.pack_into("<f", h, 80, laser)
    return bytes(h)


def acquisition_block(velocity=2.5317, size=56):
    b = bytearray(max(size, 52))
    struct.pack_into("<f", b, 48, velocity)
    return bytes(b[:size])


def spectrum_header(title, centiseconds):
    h = bytearray(96)
    struct.pack_into("<II", h, 0, 256, centiseconds)
    h[16 : 16 + len(title)] = title.encode("latin-1")
    return bytes(h)


def point_array(values, per_point):
    values = np.asarray(values, dtype="<f4").reshape(-1, per_point)
    return struct.pack("<3H", len(values) & 0xFFFF, per_point, 0) + values.tobytes() + b"\0\0"


def tiny_bmp(width=4, height=3):
    row = (3 * width + 3) // 4 * 4
    pixels = bytes(row * height)
    header = b"BM" + struct.pack("<IHHI", 54 + len(pixels), 0, 0, 54)
    header += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 0, 0, 0, 0)
    return header + pixels


def marker_bmp(width=4, height=3):
    """A bottom-up BMP whose displayed top-left pixel is red and displayed bottom-right pixel is blue, the rest black."""
    row = (3 * width + 3) // 4 * 4
    pixels = bytearray(row * height)
    pixels[(height - 1) * row : (height - 1) * row + 3] = bytes([0, 0, 255])  # last stored row = top; BGR red
    pixels[width * 3 - 3 : width * 3] = bytes([255, 0, 0])  # first stored row = bottom; BGR blue
    header = b"BM" + struct.pack("<IHHI", 54 + len(pixels), 0, 0, 54)
    header += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 0, 0, 0, 0)
    return header + bytes(pixels)


def map_definition(
    kind,
    x_range,
    y_range,
    step,
    title="synthetic map",
    *,
    x_step=None,
    y_step=None,
    n_x=7,
    n_y=9,
    file_name="synthetic.map",
    directory="C:\\synthetic\\",
    size=1188,
):
    """The key-0x82 block (docs/site/file_format.md, section 5). `kind` is a name or a raw uint16 code. The two
    steps and the two counts differ from each other and from `step`."""
    b = bytearray(max(size, 1188))
    struct.pack_into("<4H", b, 0, 0x8D, 1148, 0, MAP_KIND_CODES.get(kind, kind))
    struct.pack_into("<4d", b, 8, *x_range, *y_range)
    struct.pack_into("<d", b, 88, step)
    struct.pack_into("<2d", b, 112, 2 * step if x_step is None else x_step, 3 * step if y_step is None else y_step)
    struct.pack_into("<2I", b, 128, n_x, n_y)
    b[180 : 180 + len(title)] = title.encode("latin-1")
    b[586 : 586 + len(file_name)] = file_name.encode("latin-1")
    b[844 : 844 + len(directory)] = directory.encode("latin-1")
    return bytes(b[:size])


def map_display(first, last, x_step, y_step, *, labels=(b"Position", b"Position"), size=264):
    """The key-0x145 block: six float32 (y0, y1, dy, x0, x1, dx) at +66 and the two axis labels at +156 and +222."""
    b = bytearray(max(size, 264))
    struct.pack_into("<6f", b, 66, first[1], last[1], y_step, first[0], last[0], x_step)
    b[156 : 156 + len(labels[0])] = labels[0]
    b[222 : 222 + len(labels[1])] = labels[1]
    return bytes(b[:size])


def build_blocks(
    spectra,
    positions,
    *,
    layout="extended",
    first_x=650.0,
    last_x=4000.0,
    titles=None,
    data_kind="absorbance",
    kind=None,
    images=(),
    history="",
    with_spectra=True,
    z=None,
    image_bmp=None,
    start_centiseconds=8878,
    header_fields=None,
    definition_fields=None,
    description=None,
):
    """The directory entries (key, index, content) of a map. `images` is a list of (x0, y0, x1, y1) rectangles, each
    given a tiny BMP video image (`image_bmp`, default a black one). `z` is written as the classic layout's stage-Z
    block. `header_fields` and `definition_fields` override fields of `spectral_header` and `map_definition`;
    `description` is (first_location, last_location, x_step, y_step) of a key-0x145 block."""
    spectra = np.asarray(spectra, dtype=np.float32)
    n, n_wn = spectra.shape
    titles = titles or [f"Spectrum {i + 1} of {n}" for i in range(n)]
    header = spectral_header(n_wn, first_x, last_x, data_kind, **(header_fields or {}))
    bkg_header = spectral_header(n_wn, first_x, last_x, "single_beam")
    background = np.linspace(1.0, 2.0, n_wn, dtype="<f4").tobytes()
    blocks = [(2, 1, header), (0x6A, 1, acquisition_block()), (6, 1, point_array(positions, 2))]
    if images:
        blocks.append((6, 2, point_array(images, 4)))
    blocks.append((6, 3, point_array(np.zeros(n), 1)))
    if z is not None:
        blocks.append((6, 4, point_array(z, 1)))
    if kind:
        p = np.asarray(positions, dtype=float)
        definition = map_definition(kind, (p[0, 0], p[-1, 0]), (p[0, 1], p[-1, 1]), 25.0, **(definition_fields or {}))
        blocks.append((0x82, 0, definition))
    if description:
        blocks.append((0x145, 1, map_display(*description)))
    for i in range(len(images)):
        bmp = image_bmp or tiny_bmp()
        blocks.append((0x93, i + 1, struct.pack("<I", len(bmp)) + bmp))
    if history:
        blocks.append((0x1B, 1, history.encode("ascii") + b"\r\n\0"))
    blocks += [(2, 0, bkg_header), (0x6A, 0, acquisition_block()), (3, 0, background)]
    title_block = bytearray(264)
    title_block[0:12] = b"background 1"
    struct.pack_into("<I", title_block, 256, int((dt.datetime(2017, 1, 11, 14, 43, 28) - EPOCH).total_seconds()))
    if layout == "extended":
        blocks.append((0x6B, 0, bytes(title_block)))
    heads = [spectrum_header(t, (start_centiseconds + 9000 * i) % 2**32) for i, t in enumerate(titles)]
    if with_spectra and layout == "extended":
        series = header + bytes(56)
        for h, s in zip(heads, spectra, strict=True):
            series += h + bytes(4) + s.astype("<f4").tobytes()
        blocks.append((0x12D, 1, series))
    elif with_spectra:
        for i, (h, s) in enumerate(zip(heads, spectra, strict=True)):
            blocks += [(0x94, i + 1, h), (3, i + 1, s.astype("<f4").tobytes())]
    return blocks


def write_map(path, spectra, positions, *, layout="extended", **options):
    """Write a map; the options are those of `build_blocks`."""
    _write(path, layout, build_blocks(spectra, positions, layout=layout, **options))


def _write(path, layout, blocks):
    magic, fmt = LAYOUTS[layout]
    entry = struct.calcsize(fmt)
    start = 0x130 + entry * len(blocks)
    head = bytearray(0x130)
    head[0:18] = magic
    head[0x1E : 0x1E + 14] = b"synthetic file"
    struct.pack_into("<HI", head, 0x126, len(blocks), int((dt.datetime(2017, 1, 11, 15) - EPOCH).total_seconds()))
    directory, body = b"", b""
    for key, index, content in blocks:
        directory += struct.pack(fmt, key, start + len(body), len(content), index, 0)
        body += content
    with open(path, "wb") as f:
        f.write(bytes(head) + directory + body)
