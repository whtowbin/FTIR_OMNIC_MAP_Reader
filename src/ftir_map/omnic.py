"""Read Thermo OMNIC .map files by walking the file's block directory.

`read_omnic(path)` returns an `OmnicMap`: spectra, wavenumber axis, stage positions, map kind, backgrounds, video or
mosaic images with their stage extents, and the collection metadata that could be decoded. The file layout, the block
table and which fields are verified are described in docs/file_format.md.

This module needs only numpy and the standard library. It does not use the PyMca-derived readers.
"""

from __future__ import annotations

import datetime as _dt
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ container layout
MAGIC_CLASSIC = b"Spectral Data File"  # 16-byte directory entries, 32-bit offsets (also .spa files)
MAGIC_EXTENDED = b"Spectral Exte File"  # 22-byte directory entries, 64-bit offsets
ENTRY_FORMATS = {MAGIC_CLASSIC: ("classic", "<HIIIH"), MAGIC_EXTENDED: ("extended", "<HQIII")}
N_ENTRIES_OFFSET = 0x126  # uint16 number of directory entries
FILE_TIME_OFFSET = 0x128  # uint32 seconds since 1899-12-31
TITLE_OFFSET = 0x1E  # null-terminated file title
DIRECTORY_OFFSET = 0x130
OMNIC_EPOCH = _dt.datetime(1899, 12, 31)

# block keys
KEY_SPECTRAL_HEADER = 0x02  # 140 bytes
KEY_DATA = 0x03  # float32 spectrum
KEY_POINT_ARRAYS = 0x06  # per-point float32 arrays (positions, image rectangles, Z)
KEY_HISTORY = 0x1B  # text
KEY_ACQUISITION = 0x6A  # 56 bytes, partly decoded
KEY_BACKGROUND_TITLE = 0x6B  # 256-byte title + uint32 time
KEY_MAP_DEFINITION = 0x82
KEY_VIDEO_IMAGE = 0x93  # uint32 length + BMP file
KEY_SPECTRUM_HEADER = 0x94  # classic layout: 96-byte header in front of each spectrum
KEY_SERIES = 0x12D  # extended layout: all spectra of the map
KEY_MAP_DISPLAY = 0x145  # map title, axis labels and the map-description floats PyMca reads
KEY_MOSAIC_TILE = 0x14B  # uint32 length + JPEG file

SPECTRAL_HEADER_SIZE = 140
SPECTRUM_HEADER_SIZE = 96  # title at +16, uint32 time (centiseconds) at +4
SERIES_PREFIX = 140 + 56  # extended layout: spectral header + 56-byte block in front of the first record
SERIES_RECORD_PAD = 4  # extended layout: 4 bytes between each spectrum header and its data

# point-array kinds (key 6): values per point
POINT_SCALAR, POINT_XY, POINT_RECT = 1, 2, 4

X_UNITS = {1: "cm-1", 2: "data points"}
# y-units code -> data kind; only the codes seen in the test files are mapped
DATA_KINDS = {17: "absorbance", 16: "transmittance", 15: "single_beam", 22: "interferogram"}
MAP_DEFINITION_KINDS = {1: "points", 2: "line", 3: "area"}

POSITION_TOLERANCE_UM = 0.5  # stage coordinates closer than this are the same row / column / line


class OmnicFormatError(ValueError):
    """The file is not an OMNIC map this reader understands, or it holds no spectra."""


# ------------------------------------------------------------------ data classes
@dataclass(frozen=True)
class Block:
    """One directory entry: block key, index (0 = background / series item, 1..n = per-item), offset and size."""

    key: int
    index: int
    offset: int
    size: int


@dataclass
class CollectionParameters:
    """Fields of the spectral header (key 2) and the acquisition block (key 0x6a). `raw` keeps both blocks."""

    n_points: int
    first_x: float
    last_x: float
    x_units: str
    data_kind: str
    n_scan_points: int  # interferogram points collected
    interferogram_peak_position: int
    n_sample_scans: int
    n_fft_points: int
    n_background_scans: int
    background_gain: float
    laser_frequency: float  # cm-1
    spectrum_set_identifier: float  # PyMca's "Identifier for start indices of spectra" (+72); meaning unverified
    mirror_velocity: float | None  # cm/s; INFERRED (key 0x6a +48 holds standard Nicolet velocities such as 2.5317)
    raw: dict[str, bytes] = field(default_factory=dict, repr=False)

    @property
    def data_spacing(self) -> float:
        return (self.last_x - self.first_x) / (self.n_points - 1) if self.n_points > 1 else 0.0


@dataclass
class Background:
    """A background single-beam spectrum. Extended files store one per re-collection during the map."""

    spectrum: np.ndarray  # float32 as stored
    wavenumber: np.ndarray
    data_kind: str
    title: str = ""
    time: _dt.datetime | None = None  # from the title block; None in the classic layout


@dataclass
class StageImage:
    """A video image (BMP) or mosaic tile (JPEG) with the stage rectangle it covers."""

    index: int
    format: str  # "BMP" or "JPEG"
    data: bytes = field(repr=False)  # the embedded image file
    width: int | None
    height: int | None
    extent: tuple[float, float, float, float] | None  # (x0, y0, x1, y1) stage um; image orientation UNVERIFIED

    @property
    def pixel_size_um(self) -> tuple[float, float] | None:
        if self.extent is None or not self.width or not self.height:
            return None
        x0, y0, x1, y1 = self.extent
        return abs(x1 - x0) / self.width, abs(y1 - y0) / self.height


@dataclass
class MapDefinition:
    """The map set-up stored in block 0x82. Field meanings were inferred by comparing 81 files."""

    kind: str  # "points", "line" or "area" (uint16 at +6), or "unknown (n)"
    x_range: tuple[float, float]  # area: first/last column centre; line: start/end; points: bounding box
    y_range: tuple[float, float]
    step: float  # line step (also set for area maps)
    x_step: float  # area maps; may be left over from an earlier map on line and point maps
    y_step: float
    n_x: int  # area maps; may be left over from an earlier map on line and point maps
    n_y: int
    title: str
    file_name: str
    directory: str


@dataclass
class MapDescription:
    """The map-description floats in block 0x145 that PyMca reads (first/last location and step)."""

    first_location: tuple[float, float]  # (x, y)
    last_location: tuple[float, float]
    x_step: float
    y_step: float


@dataclass
class OmnicMap:
    """Contents of an OMNIC .map file. Arrays are in file (acquisition) order."""

    path: Path
    layout: str  # "classic" or "extended"
    wavenumber: np.ndarray  # (n_wn,) float64; ascending for cm-1 data; for interferograms the stored point index
    spectra: np.ndarray  # (n_points, n_wn) float32 as stored (non-finite values kept)
    x: np.ndarray  # (n_points,) stage X in um (float32 values as stored, held as float64); NaN if absent
    y: np.ndarray
    z: np.ndarray | None  # stage Z (focus) in um; recorded by the classic layout (point maps) only
    map_kind: str  # "grid", "line", "points" or "array"
    grid_shape: tuple[int, int] | None  # (n_rows, n_columns) when the positions fill a regular grid
    grid_index: tuple[np.ndarray, np.ndarray] | None  # (row, column) of every point when gridded
    titles: list[str]
    times_s: np.ndarray  # seconds since the first spectrum (from a centisecond counter in each spectrum header)
    data_kind: str  # "absorbance", "transmittance", "single_beam", "interferogram" or "unknown (code n)"
    collection: CollectionParameters
    backgrounds: list[Background]
    background_index: np.ndarray | None  # (n_points,) which entry of `backgrounds` each spectrum was ratioed against
    images: list[StageImage]
    map_definition: MapDefinition | None
    map_description: MapDescription | None
    history: str
    title: str
    file_time: _dt.datetime
    blocks: list[Block] = field(repr=False)

    @property
    def n_points(self) -> int:
        return self.spectra.shape[0]

    def block_bytes(self, key: int, index: int | None = None) -> list[bytes]:
        """Raw contents of the blocks with this key (and index), for fields this reader does not decode."""
        data = self.path.read_bytes()
        return [data[b.offset : b.offset + b.size] for b in self.blocks if b.key == key and index in (None, b.index)]

    def spectra_grid(self) -> np.ndarray:
        """Spectra as (n_rows, n_columns, n_wn), NaN where a grid cell has no spectrum. Grid maps only."""
        if self.grid_shape is None or self.grid_index is None:
            raise ValueError(f"{self.path.name}: a {self.map_kind} map has no regular grid")
        out = np.full((*self.grid_shape, self.spectra.shape[1]), np.nan, dtype=self.spectra.dtype)
        out[self.grid_index] = self.spectra
        return out


# ------------------------------------------------------------------ low-level readers
def read_directory(data: bytes) -> tuple[str, list[Block]]:
    """Layout name and the list of directory entries."""
    magic = data[:18]
    if magic not in ENTRY_FORMATS:
        raise OmnicFormatError(f"not an OMNIC container (first bytes {magic!r})")
    layout, fmt = ENTRY_FORMATS[magic]
    step = struct.calcsize(fmt)
    (n_entries,) = struct.unpack_from("<H", data, N_ENTRIES_OFFSET)
    blocks = []
    for i in range(n_entries):
        key, offset, size, index, _ = struct.unpack_from(fmt, data, DIRECTORY_OFFSET + i * step)
        if offset + size > len(data):
            raise OmnicFormatError(f"directory entry {i} (key {key:#x}) points past the end of the file")
        blocks.append(Block(key, index, offset, size))
    return layout, blocks


def _find(blocks: list[Block], key: int, index: int | None = None) -> list[Block]:
    return [b for b in blocks if b.key == key and index in (None, b.index)]


def _cstring(data: bytes, offset: int, max_len: int) -> str:
    return data[offset : offset + max_len].split(b"\0", 1)[0].decode("latin-1")


def _data_kind(y_units: int) -> str:
    return DATA_KINDS.get(y_units, f"unknown (code {y_units})")


def _axis(header: CollectionParameters) -> np.ndarray:
    return np.linspace(header.first_x, header.last_x, header.n_points)


def parse_spectral_header(data: bytes, offset: int, acquisition: bytes = b"") -> CollectionParameters:
    """Key-2 block (140 bytes). Offsets follow PyMca's _getOmnicInfo; the units codes were checked on 81 files."""
    u = struct.unpack_from("<35I", data, offset)
    f = struct.unpack_from("<31f", data, offset + 16)
    velocity = struct.unpack_from("<f", acquisition, 48)[0] if len(acquisition) >= 52 else None
    raw = {"spectral_header": bytes(data[offset : offset + SPECTRAL_HEADER_SIZE]), "acquisition": acquisition}
    return CollectionParameters(
        n_points=u[1],
        first_x=f[1],
        last_x=f[0],
        x_units=X_UNITS.get(u[2], f"unknown (code {u[2]})"),
        data_kind=_data_kind(u[3]),
        n_scan_points=u[7],
        interferogram_peak_position=u[8],
        n_sample_scans=u[9],
        n_fft_points=u[11],
        n_background_scans=u[13],
        background_gain=f[10],
        laser_frequency=f[16],
        spectrum_set_identifier=f[14],
        mirror_velocity=velocity,
        raw=raw,
    )


def parse_point_arrays(data: bytes, blocks: list[Block]) -> list[tuple[int, int, np.ndarray]]:
    """Key-6 blocks as (index, values per point, float64 array of shape (count, values per point))."""
    out = []
    for b in _find(blocks, KEY_POINT_ARRAYS):
        if b.size < 6:
            continue
        count, kind, _ = struct.unpack_from("<3H", data, b.offset)
        if kind == 0 or 6 + 4 * count * kind > b.size:
            continue
        values = np.frombuffer(data, "<f4", count * kind, b.offset + 6).astype(np.float64)
        out.append((b.index, kind, values.reshape(count, kind)))
    return out


def _spectra_extended(
    data: bytes, blocks: list[Block], n_wn: int
) -> tuple[list[np.ndarray], list[bytes], CollectionParameters]:
    series = _find(blocks, KEY_SERIES)
    if not series:
        raise OmnicFormatError("no spectra block (key 0x12d): a mosaic-only or empty map")
    s = series[0]
    header = parse_spectral_header(data, s.offset)
    if header.n_points != n_wn:
        raise OmnicFormatError(f"the spectra block has {header.n_points} points per spectrum, the file header {n_wn}")
    record = SPECTRUM_HEADER_SIZE + SERIES_RECORD_PAD + 4 * n_wn
    n, rest = divmod(s.size - SERIES_PREFIX, record)
    if rest or n < 1:
        raise OmnicFormatError(f"spectra block of {s.size} bytes is not {SERIES_PREFIX} + n x {record}")
    starts = [s.offset + SERIES_PREFIX + i * record for i in range(n)]
    heads = [data[o : o + SPECTRUM_HEADER_SIZE] for o in starts]
    spectra = [np.frombuffer(data, "<f4", n_wn, o + SPECTRUM_HEADER_SIZE + SERIES_RECORD_PAD) for o in starts]
    return spectra, heads, header


def _spectra_classic(data: bytes, blocks: list[Block], n_wn: int) -> tuple[list[np.ndarray], list[bytes]]:
    heads = {b.index: b for b in _find(blocks, KEY_SPECTRUM_HEADER)}
    spectra, head_bytes = [], []
    for b in sorted((b for b in _find(blocks, KEY_DATA) if b.index >= 1), key=lambda b: b.index):
        if b.size != 4 * n_wn:
            raise OmnicFormatError(f"spectrum {b.index} has {b.size} bytes, expected {4 * n_wn}")
        h = heads.get(b.index)
        head_bytes.append(data[h.offset : h.offset + SPECTRUM_HEADER_SIZE] if h else bytes(SPECTRUM_HEADER_SIZE))
        spectra.append(np.frombuffer(data, "<f4", n_wn, b.offset))
    if not spectra:
        raise OmnicFormatError("no spectra (no key-3 blocks with index >= 1)")
    return spectra, head_bytes


def _times_s(heads: list[bytes]) -> np.ndarray:
    """Seconds since the first spectrum. The uint32 at +4 counts centiseconds; differences are taken modulo 2**32."""
    ticks = np.array([struct.unpack_from("<I", h, 4)[0] for h in heads], dtype=np.int64)
    return ((ticks - ticks[0]) % 2**32) / 100.0


def _backgrounds(data: bytes, blocks: list[Block]) -> list[Background]:
    """Background spectra: key-3 index-0 blocks, each described by the key-2 index-0 block before it and titled by
    the key-0x6b block after it (extended layout)."""
    out: list[Background] = []
    header = None
    for b in blocks:
        if b.key == KEY_SPECTRAL_HEADER and b.index == 0:
            header = parse_spectral_header(data, b.offset)
        elif b.key == KEY_DATA and b.index == 0 and header is not None and b.size == 4 * header.n_points:
            spectrum = np.frombuffer(data, "<f4", header.n_points, b.offset).copy()
            out.append(Background(spectrum, _axis(header), header.data_kind))
        elif b.key == KEY_BACKGROUND_TITLE and out and b.size >= 260:
            out[-1].title = _cstring(data, b.offset, 256)
            out[-1].time = OMNIC_EPOCH + _dt.timedelta(seconds=struct.unpack_from("<I", data, b.offset + 256)[0])
    return out


def _image_size(fmt: str, img: bytes) -> tuple[int | None, int | None]:
    if fmt == "BMP" and len(img) >= 26:
        w, h = struct.unpack_from("<ii", img, 18)
        return w, abs(h)
    i = 2
    while fmt == "JPEG" and i + 9 < len(img):  # walk the JPEG segments to the start-of-frame marker
        if img[i] != 0xFF:
            break
        marker, length = img[i + 1], struct.unpack_from(">H", img, i + 2)[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h, w = struct.unpack_from(">HH", img, i + 5)
            return w, h
        i += 2 + length
    return None, None


def _images(data: bytes, blocks: list[Block], rects: np.ndarray | None) -> list[StageImage]:
    """Mosaic tiles (key 0x14b, JPEG) when present, otherwise video images (key 0x93, BMP). The n-th image covers
    the n-th rectangle of the key-6 rectangle block (count matched in all 81 test files)."""
    tiles = _find(blocks, KEY_MOSAIC_TILE)
    chosen, fmt = (tiles, "JPEG") if tiles else (_find(blocks, KEY_VIDEO_IMAGE), "BMP")
    out = []
    for i, b in enumerate(sorted(chosen, key=lambda b: b.index)):
        (n,) = struct.unpack_from("<I", data, b.offset)
        img = bytes(data[b.offset + 4 : b.offset + 4 + min(n, b.size - 4)])
        w, h = _image_size(fmt, img)
        extent = None
        if rects is not None and i < len(rects):
            x0, y0, x1, y1 = (float(v) for v in rects[i])
            extent = (x0, y0, x1, y1)
        out.append(StageImage(b.index, fmt, img, w, h, extent))
    return out


def _map_definition(data: bytes, blocks: list[Block]) -> MapDefinition | None:
    found = _find(blocks, KEY_MAP_DEFINITION)
    if not found or found[0].size < 1100:
        return None
    o = found[0].offset
    (code,) = struct.unpack_from("<H", data, o + 6)
    x0, x1, y0, y1 = struct.unpack_from("<4d", data, o + 8)
    (step,) = struct.unpack_from("<d", data, o + 88)
    x_step, y_step = struct.unpack_from("<2d", data, o + 112)
    n_x, n_y = struct.unpack_from("<2I", data, o + 128)
    return MapDefinition(
        kind=MAP_DEFINITION_KINDS.get(code, f"unknown ({code})"),
        x_range=(x0, x1),
        y_range=(y0, y1),
        step=step,
        x_step=x_step,
        y_step=y_step,
        n_x=n_x,
        n_y=n_y,
        title=_cstring(data, o + 180, 256),
        file_name=_cstring(data, o + 586, 256),
        directory=_cstring(data, o + 844, 256),
    )


def _map_description(data: bytes, blocks: list[Block]) -> MapDescription | None:
    """Six float32 (y0, y1, dy, x0, x1, dx) at +66 of block 0x145, valid when both axis labels read "Position"."""
    found = _find(blocks, KEY_MAP_DISPLAY)
    if not found or found[0].size < 240:
        return None
    o = found[0].offset
    if not (data[o + 156 : o + 164] == b"Position" and data[o + 222 : o + 230] == b"Position"):
        return None
    y0, y1, dy, x0, x1, dx = struct.unpack_from("<6f", data, o + 66)
    return MapDescription((x0, y0), (x1, y1), dx, dy)


# ------------------------------------------------------------------ map geometry
def _clusters(v: np.ndarray, tol: float) -> np.ndarray:
    """Rank of each value among the distinct values (values closer than tol are one value)."""
    order = np.argsort(v, kind="stable")
    new = np.concatenate([[True], np.diff(v[order]) > tol])
    rank = np.empty(len(v), dtype=np.int64)
    rank[order] = np.cumsum(new) - 1
    return rank


def grid_layout(x: np.ndarray, y: np.ndarray, tol: float = POSITION_TOLERANCE_UM):
    """((n_rows, n_columns), (row, column)) when every grid cell holds exactly one point, else None."""
    if len(x) < 4 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    col, row = _clusters(x, tol), _clusters(y, tol)
    shape = (int(row.max()) + 1, int(col.max()) + 1)
    if shape[0] < 2 or shape[1] < 2 or shape[0] * shape[1] != len(x):
        return None
    if len(set(zip(row.tolist(), col.tolist(), strict=True))) != len(x):
        return None
    return shape, (row, col)


def is_collinear(x: np.ndarray, y: np.ndarray, tol: float = POSITION_TOLERANCE_UM) -> bool:
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return False
    p = np.column_stack([x, y]) - [x.mean(), y.mean()]
    if np.ptp(p, axis=0).max() <= tol:
        return False
    direction = np.linalg.svd(p, full_matrices=False)[2][0]
    off_line = p @ np.array([-direction[1], direction[0]])
    return bool(np.max(np.abs(off_line)) <= tol)


def classify(x, y, titles, definition: MapDefinition | None):
    """(map_kind, grid layout or None). Array-detector maps title their spectra "Pixel ..."."""
    layout = grid_layout(x, y)
    if titles and all(t.startswith("Pixel ") for t in titles):
        return "array", layout
    if definition is not None and definition.kind == "points":
        return "points", None
    if layout is not None:
        return "grid", layout
    if is_collinear(x, y):
        return "line", None
    return "points", None


def _positions(point_arrays, n: int):
    """(x, y, z, background_index) from the key-6 blocks. Index 1 holds one (x, y) pair per spectrum. Index 3 holds,
    per spectrum, the number of the background it was ratioed against (its maximum is the number of backgrounds
    minus one in every test file). Index 4, written by the classic layout only, holds the stage Z."""
    by_index = {(index, kind): a for (index, kind, a) in point_arrays if len(a) == n}
    xy = by_index.get((1, POINT_XY))
    x, y = (xy[:, 0], xy[:, 1]) if xy is not None else (np.full(n, np.nan), np.full(n, np.nan))
    bkg = by_index.get((3, POINT_SCALAR))
    z = by_index.get((4, POINT_SCALAR))
    return x, y, (z[:, 0] if z is not None else None), (bkg[:, 0].astype(np.int64) if bkg is not None else None)


# ------------------------------------------------------------------ entry point
def read_omnic(path) -> OmnicMap:
    """Read an OMNIC .map file. Raises OmnicFormatError on files that are not OMNIC maps or hold no spectra."""
    path = Path(path)
    data = path.read_bytes()
    layout, blocks = read_directory(data)
    main = _find(blocks, KEY_SPECTRAL_HEADER, 1) or _find(blocks, KEY_SPECTRAL_HEADER)
    if not main:
        raise OmnicFormatError(f"{path.name}: no spectral header (key 2)")
    acq = _find(blocks, KEY_ACQUISITION, 1)
    acquisition = bytes(data[acq[0].offset : acq[0].offset + acq[0].size]) if acq else b""
    collection = parse_spectral_header(data, main[0].offset, acquisition)
    n_wn = collection.n_points
    try:
        if layout == "extended":
            spectra, heads, series_header = _spectra_extended(data, blocks, n_wn)
            data_kind = series_header.data_kind
        else:
            spectra, heads = _spectra_classic(data, blocks, n_wn)
            data_kind = collection.data_kind
    except OmnicFormatError as e:
        raise OmnicFormatError(f"{path.name}: {e}") from None
    spectra = np.vstack(spectra).astype(np.float32)
    wavenumber = _axis(collection)
    if collection.x_units == "cm-1" and wavenumber[0] > wavenumber[-1]:
        wavenumber, spectra = wavenumber[::-1].copy(), spectra[:, ::-1].copy()

    titles = [_cstring(h, 16, 64) for h in heads]
    point_arrays = parse_point_arrays(data, blocks)
    x, y, z, background_index = _positions(point_arrays, len(spectra))
    definition = _map_definition(data, blocks)
    map_kind, grid = classify(x, y, titles, definition)
    rects = next((a for (_, kind, a) in point_arrays if kind == POINT_RECT), None)
    history = "\n".join(_cstring(data, b.offset, b.size).strip() for b in _find(blocks, KEY_HISTORY))
    (secs,) = struct.unpack_from("<I", data, FILE_TIME_OFFSET)
    return OmnicMap(
        path=path,
        layout=layout,
        wavenumber=wavenumber,
        spectra=spectra,
        x=x,
        y=y,
        z=z,
        map_kind=map_kind,
        grid_shape=grid[0] if grid else None,
        grid_index=grid[1] if grid else None,
        titles=titles,
        times_s=_times_s(heads),
        data_kind=data_kind,
        collection=collection,
        backgrounds=_backgrounds(data, blocks),
        background_index=background_index,
        images=_images(data, blocks, rects),
        map_definition=definition,
        map_description=_map_description(data, blocks),
        history=history,
        title=_cstring(data, TITLE_OFFSET, 256),
        file_time=OMNIC_EPOCH + _dt.timedelta(seconds=secs),
        blocks=blocks,
    )
