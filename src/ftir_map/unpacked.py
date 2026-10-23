"""Unpack one map into an open folder, and read it back.

`unpack_map(m, folder)` (also `m.unpack(folder)`) writes `map.json`, `spectra.zarr`, `points.csv`, `images/` and a
`README.txt`. Nothing in the folder needs this package or OMNIC to read. `read_unpacked(folder)` rebuilds the
`OmnicMap` from those files. The layout is specified in docs/unpacked_format.md.

The spectra, axis, positions, titles, times, backgrounds, images and decoded metadata round-trip exactly. The raw
header bytes (`collection.raw`) and the block directory (`blocks`) are not kept, so `OmnicMap.block_bytes` needs the
original .map file.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .exports import (
    axis_dimension,
    axis_name,
    backgrounds_array,
    build_metadata,
    dataset_to_zarr,
    import_optional,
    reader_version,
)
from .omnic import (
    Background,
    CollectionParameters,
    MapDefinition,
    MapDescription,
    OmnicFormatError,
    OmnicMap,
    StageImage,
)
from .writers import save_images, save_points, save_spectra, write_json

FORMAT_NAME = "ftir-map unpacked"
FORMAT_VERSION = 1
MAP_JSON = "map.json"
SPECTRA_FORMATS = ("zarr", "csv", "nc")  # nc is an opt-in export that needs the [netcdf] extra
DEFAULT_SPECTRA = "zarr"
SPECTRA_FILES = {"nc": "spectra.nc", "zarr": "spectra.zarr", "csv": "spectra.csv"}
BACKGROUNDS_CSV = "backgrounds.csv"
POINTS_CSV = "points.csv"
IMAGES_DIR = "images"
SOURCE_DIR = "source"
README = "README.txt"
# everything unpack_map writes; an overwrite removes these names and nothing else
OWN_NAMES = (MAP_JSON, README, POINTS_CSV, BACKGROUNDS_CSV, IMAGES_DIR, SOURCE_DIR, *SPECTRA_FILES.values())

COLLECTION_FLOATS = ("first_x", "last_x", "background_gain", "laser_frequency", "spectrum_set_identifier")
DEFINITION_FLOATS = ("step", "x_step", "y_step")
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")  # writers.csv_cell puts a quote in front of these
# (variable, points.csv column, dtype) for the variables a map may not have
CSV_OPTIONAL_COLUMNS = (
    ("z", "z_um", np.float64),
    ("grid_row", "row", np.int64),
    ("grid_column", "column", np.int64),
    ("background_index", "background_index", np.int64),
)
NC_COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ the spectra dataset
def point_dataset(m: OmnicMap) -> xr.Dataset:
    """Every map as (point, axis) in file order, with the per-point arrays beside the spectra.

    Unlike `to_xarray`, grid maps are not reshaped: `grid_row` and `grid_column` say where each point belongs, and
    titles are kept. Variables: spectra, x, y, z (if recorded), grid_row and grid_column (grid maps), title, time_s,
    background_index (if recorded) and backgrounds (if there are any).
    """
    dim = axis_name(m)
    variables: dict[str, tuple[Any, ...]] = {
        "spectra": (("point", dim), m.spectra, {"long_name": m.data_kind}),
        "x": ("point", m.x, {"units": "um", "long_name": "stage X"}),
        "y": ("point", m.y, {"units": "um", "long_name": "stage Y"}),
    }
    if m.z is not None:
        variables["z"] = ("point", m.z, {"units": "um", "long_name": "stage Z"})
    if m.grid_index is not None:
        variables["grid_row"] = ("point", m.grid_index[0], {"long_name": "row of the grid cell, from 0"})
        variables["grid_column"] = ("point", m.grid_index[1], {"long_name": "column of the grid cell, from 0"})
    variables["title"] = ("point", np.array(m.titles, dtype=str), {"long_name": "spectrum title"})
    variables["time_s"] = ("point", m.times_s, {"long_name": "time since the first spectrum, in seconds"})
    if m.background_index is not None:
        variables["background_index"] = ("point", m.background_index, {"long_name": "row of backgrounds used"})
    backgrounds = backgrounds_array(m)
    if backgrounds is not None:
        variables["backgrounds"] = (("background", dim), backgrounds, {"long_name": "background single-beam"})
    attrs = {"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "source_file": m.path.name}
    coords = {"point": np.arange(m.n_points), dim: (dim, m.axis, {"units": m.axis_unit})}
    return xr.Dataset(variables, coords=coords, attrs=attrs)


def _netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """No fill value on floats (NaN stays NaN, as stored); the two big arrays are compressed."""
    encoding: dict[str, dict[str, Any]] = {
        str(name): {"_FillValue": None} for name, var in dataset.variables.items() if var.dtype.kind == "f"
    }
    for name in ("spectra", "backgrounds"):
        if name in dataset:
            encoding[name].update(NC_COMPRESSION)
    return encoding


def _write_backgrounds_csv(m: OmnicMap, path: Path) -> None:
    """The spectral axis, then one column per background named by its index (the layout of spectra.csv)."""
    backgrounds = backgrounds_array(m)
    if backgrounds is None:
        return
    header = [f"{axis_name(m)}_{m.axis_unit}", *map(str, range(len(backgrounds)))]
    table = np.column_stack([m.axis, backgrounds.T])
    np.savetxt(path, table, fmt="%.9g", delimiter=",", header=",".join(header), comments="")


def _write_spectra(m: OmnicMap, folder: Path, spectra: str) -> str:
    path = folder / SPECTRA_FILES[spectra]
    if spectra == "csv":
        save_spectra(m, path, "csv")
        _write_backgrounds_csv(m, folder / BACKGROUNDS_CSV)
    elif spectra == "zarr":
        dataset_to_zarr(point_dataset(m), path, "w")
    else:
        import_optional("h5netcdf", "netcdf", "netCDF output")
        dataset = point_dataset(m)
        dataset.to_netcdf(path, engine="h5netcdf", encoding=_netcdf_encoding(dataset))
    return path.name


# ------------------------------------------------------------------ writing
def _background_records(m: OmnicMap) -> list[dict[str, Any]]:
    """Per-background fields that the spectra file does not hold. Empty if the backgrounds are not stored."""
    if backgrounds_array(m) is None:
        if m.backgrounds:
            warnings.warn(
                f"{m.path.name}: backgrounds with another length than the spectra are not stored", stacklevel=3
            )
        return []
    return [
        {
            "title": b.title,
            "time": b.time.isoformat() if b.time else None,
            "data_kind": b.data_kind,
            "wavenumber_first": float(b.wavenumber[0]),
            "wavenumber_last": float(b.wavenumber[-1]),
        }
        for b in m.backgrounds
    ]


def _image_records(m: OmnicMap, folder: Path, png: bool) -> list[dict[str, Any]]:
    """Write the embedded images (and PNG copies if asked); return what read_unpacked needs to rebuild them."""
    written = save_images(m, folder / IMAGES_DIR, "original")
    if png and written:
        save_images(m, folder / IMAGES_DIR / "png", "png")
    return [
        {
            "index": s.index,
            "format": s.format,
            "file": w["file"],
            "width": s.width,
            "height": s.height,
            "extent": s.extent,
        }
        for s, w in zip(m.images, written, strict=True)
    ]


def _readme_text(m: OmnicMap, spectra_file: str, spectra: str) -> str:
    axis = axis_name(m)
    lines = [
        f"{m.path.name}: {m.map_kind} map, {m.n_points} spectra, unpacked by ftir-OMNIC-map-reader ({FORMAT_NAME}).",
        "",
        "Nothing here needs OMNIC or the ftir-map package. The files:",
        "  map.json      metadata as JSON: source file, collection parameters, map definition, history, backgrounds",
        "  points.csv    one row per spectrum: stage position (um), grid row and column, title, time, background",
        f"  {spectra_file:<13} the spectra ({axis} axis in {m.axis_unit}), in the order they were collected",
        "  images/       the video images or mosaic tiles as OMNIC stored them (BMP or JPEG), if the map has any,",
        "                with world files, images.csv and images.json (stage rectangle of each); png/ if requested",
        "  source/       a copy of the original .map file, only if it was requested",
        "",
    ]
    lines += {"nc": _NC_HELP, "zarr": _ZARR_HELP, "csv": _CSV_HELP}[spectra]
    lines += ["", "Positions are stage coordinates in um. Format details: docs/unpacked_format.md in the package."]
    return "\n".join(lines) + "\n"


_NC_HELP = [
    f"Open {SPECTRA_FILES['nc']} (netCDF4, which is HDF5) without this package:",
    "  Python   import xarray as xr; ds = xr.open_dataset('spectra.nc', engine='h5netcdf')",
    "           (or h5py.File('spectra.nc')['spectra'][:])",
    "  MATLAB   spectra = ncread('spectra.nc', 'spectra');  % size is [axis points]: MATLAB lists dimensions reversed",
    "  R        nc <- ncdf4::nc_open('spectra.nc'); spectra <- ncdf4::ncvar_get(nc, 'spectra')",
    "  Julia    using NCDatasets; ds = NCDataset('spectra.nc'); ds['spectra'][:, :]",
    "Variables: spectra (point, axis), the axis itself with its unit, x, y, z, grid_row, grid_column, title, time_s,",
    "background_index, and backgrounds (background, axis). Variables that the map did not record are absent.",
]
_ZARR_HELP = [
    "Open spectra.zarr (a folder) without this package:",
    "  Python   import xarray as xr; ds = xr.open_zarr('spectra.zarr')   (or zarr.open('spectra.zarr'))",
    "  R        Rarr::read_zarr_array('spectra.zarr/spectra')   (Rarr package)",
    "  Julia    using Zarr; z = zopen('spectra.zarr'); z['spectra'][:, :]",
    "  MATLAB or Excel: unpack again with --spectra csv.",
    "Variables: spectra (point, axis), the axis itself with its unit, x, y, z, grid_row, grid_column, title, time_s,",
    "background_index, and backgrounds (background, axis). Variables that the map did not record are absent.",
]
_CSV_HELP = [
    "spectra.csv: the first column is the spectral axis, then one column per spectrum named by its index (the row",
    "number in points.csv). backgrounds.csv has the same layout. Values keep 9 significant digits, so the axis is",
    "rounded; the spectra (float32) are exact. points.csv holds the positions and titles.",
]


def contained(base: str | Path, relative: str | Path) -> Path:
    """`base / relative`, refusing any path that leaves `base`.

    Unpacked folders and projects are meant to be shared, so file names read from map.json or catalog.json are
    untrusted: a name such as "../../x" must not make the reader open, or the writer clear, anything outside.
    """
    root = Path(base).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"{str(relative)!r} points outside {base}")
    return target


def _clear_previous(folder: Path) -> None:
    """Remove the files this package writes, before an overwrite.

    A symbolic link (dangling or not) is removed as a link and never followed, so every file is then written fresh
    inside `folder`; a shared folder whose spectra.zarr or images/ points elsewhere cannot redirect the writes.
    """
    for name in OWN_NAMES:
        target = folder / name
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def unpack_map(
    m: OmnicMap,
    folder: str | Path,
    *,
    spectra: str = DEFAULT_SPECTRA,
    png: bool = False,
    copy_source: bool = False,
    overwrite: bool = False,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Write `m` as an unpacked-map folder (docs/unpacked_format.md) and return the content of its `map.json`.

    `spectra` is "zarr" (default, writes spectra.zarr), "csv", or "nc" (netCDF4, needs the [netcdf] extra). `png` adds PNG copies of the images in `images/png/`, and
    `copy_source` copies the original .map file into `source/`. `map.json` is written last, so a folder without one
    is an unfinished write. `sha256` is the checksum of the source file, if the caller has it already.

    Raises:
        FileExistsError: if `folder` exists and is not empty and `overwrite` is False.
        ValueError: for an unsupported `spectra` format.
    """
    if spectra not in SPECTRA_FORMATS:
        raise ValueError(f"cannot write spectra as {spectra!r}; use one of {list(SPECTRA_FORMATS)}")
    folder = Path(folder)
    if folder.is_dir() and any(folder.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{folder} is not empty; pass overwrite=True to replace the unpacked files in it")
        _clear_previous(folder)
    folder.mkdir(parents=True, exist_ok=True)

    spectra_file = _write_spectra(m, folder, spectra)
    save_points(m, folder / POINTS_CSV, "csv")
    images = _image_records(m, folder, png)
    copied = None
    if copy_source:
        (folder / SOURCE_DIR).mkdir()
        copied = f"{SOURCE_DIR}/{m.path.name}"
        shutil.copy2(m.path, folder / copied)
    (folder / README).write_text(_readme_text(m, spectra_file, spectra), encoding="utf-8")
    content = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "reader_version": reader_version(),
        "unpacked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "file_name": m.path.name,
            "path": str(m.path),
            "size_bytes": m.path.stat().st_size,
            "sha256": sha256 or sha256_file(m.path),
            "copied_to": copied,
        },
        "files": {"spectra": spectra_file, "spectra_format": spectra, "points": POINTS_CSV, "images": IMAGES_DIR},
        "metadata": build_metadata(m),
        "backgrounds": _background_records(m),
        "images": images,
    }
    write_json(folder / MAP_JSON, content)
    return content


# ------------------------------------------------------------------ reading
def read_map_json(folder: Path) -> dict[str, Any]:
    """The parsed `map.json` of an unpacked-map folder, checked for format name and version."""
    path = folder / MAP_JSON
    if not path.is_file():
        raise OmnicFormatError(f"{folder}: no {MAP_JSON}; this is not an unpacked map folder")
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("format") != FORMAT_NAME:
        raise OmnicFormatError(f"{path}: format is {content.get('format')!r}, not {FORMAT_NAME!r}")
    if content.get("format_version") != FORMAT_VERSION:
        raise OmnicFormatError(
            f"{path}: format version {content.get('format_version')}, this reader reads {FORMAT_VERSION}"
        )
    return content


def _nan(value: float | None) -> float:
    return float("nan") if value is None else value


def _unescape_cell(text: str) -> str:
    """Undo the quote that `writers.csv_cell` puts before text that a spreadsheet would run as a formula."""
    return text[1:] if text.startswith("'") and text[1:2] in FORMULA_PREFIXES else text


def _points_from_csv(path: Path) -> dict[str, Any]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    def column(name: str, dtype: Any = np.float64) -> np.ndarray:
        """Empty cells (non-finite numbers, written as None) become NaN."""
        return np.array([row[name] or "nan" for row in rows], dtype=np.float64).astype(dtype)

    variables: dict[str, Any] = {"x": column("x_um"), "y": column("y_um"), "time_s": column("time_s")}
    variables["title"] = [_unescape_cell(row["title"]) for row in rows]
    for name, source, dtype in CSV_OPTIONAL_COLUMNS:
        if source in rows[0]:
            variables[name] = column(source, dtype)
    return variables


def _dataset_from_csv(folder: Path, meta: dict[str, Any]) -> xr.Dataset:
    """The `point_dataset` variables, rebuilt from spectra.csv, backgrounds.csv and points.csv."""
    dim = axis_dimension(meta["metadata"]["axis_unit"])
    table = np.loadtxt(contained(folder, meta["files"]["spectra"]), delimiter=",", skiprows=1, ndmin=2)
    points = _points_from_csv(contained(folder, meta["files"]["points"]))
    variables: dict[str, Any] = {"spectra": (("point", dim), table[:, 1:].T.astype(np.float32))}
    variables.update({name: ("point", values) for name, values in points.items()})
    if (folder / BACKGROUNDS_CSV).exists():
        backgrounds = np.loadtxt(folder / BACKGROUNDS_CSV, delimiter=",", skiprows=1, ndmin=2)
        variables["backgrounds"] = (("background", dim), backgrounds[:, 1:].T.astype(np.float32))
    return xr.Dataset(variables, coords={dim: table[:, 0]})


def _open_spectra(folder: Path, meta: dict[str, Any]) -> xr.Dataset:
    """The spectra file as a dataset held in memory; times are left as plain numbers."""
    kind, path = meta["files"]["spectra_format"], contained(folder, meta["files"]["spectra"])
    if kind == "csv":
        return _dataset_from_csv(folder, meta)
    engine = "zarr"
    if kind == "nc":
        import_optional("h5netcdf", "netcdf", "reading spectra.nc")
        engine = "h5netcdf"
    with xr.open_dataset(path, engine=engine, decode_times=False, decode_timedelta=False) as dataset:
        return dataset.load()


def _collection(values: dict[str, Any]) -> CollectionParameters:
    fields = {k: v for k, v in values.items() if k != "data_spacing"}
    fields.update({name: _nan(fields[name]) for name in COLLECTION_FLOATS})
    return CollectionParameters(**fields)


def _definition(values: dict[str, Any] | None) -> MapDefinition | None:
    if values is None:
        return None
    fields = dict(values)
    fields.update({name: _nan(fields[name]) for name in DEFINITION_FLOATS})
    fields.update({name: tuple(map(_nan, fields[name])) for name in ("x_range", "y_range")})
    return MapDefinition(**fields)


def _pair(values: list[float | None]) -> tuple[float, float]:
    return _nan(values[0]), _nan(values[1])


def _description(values: dict[str, Any] | None) -> MapDescription | None:
    if values is None:
        return None
    first, last = _pair(values["first_location"]), _pair(values["last_location"])
    return MapDescription(first, last, x_step=_nan(values["x_step"]), y_step=_nan(values["y_step"]))


def _backgrounds(dataset: xr.Dataset, records: list[dict[str, Any]], axis_unit: str) -> list[Background]:
    """Backgrounds in the order and direction they were read: the file holds them on the ascending spectra axis."""
    out = []
    for row, record in zip(dataset["backgrounds"].values if records else [], records, strict=True):
        first, last = record["wavenumber_first"], record["wavenumber_last"]
        descending = axis_unit == "cm-1" and first > last
        spectrum = np.ascontiguousarray(row[::-1] if descending else row, dtype=np.float32)
        time = dt.datetime.fromisoformat(record["time"]) if record["time"] else None
        out.append(Background(spectrum, np.linspace(first, last, len(row)), record["data_kind"], record["title"], time))
    return out


def _images(folder: Path, files: dict[str, Any], records: list[dict[str, Any]]) -> list[StageImage]:
    out = []
    for record in records:
        extent = record["extent"]
        data = contained(folder, Path(files["images"]) / record["file"]).read_bytes()
        out.append(
            StageImage(
                record["index"],
                record["format"],
                data,
                record["width"],
                record["height"],
                (_nan(extent[0]), _nan(extent[1]), _nan(extent[2]), _nan(extent[3])) if extent else None,
            )
        )
    return out


def _optional(dataset: xr.Dataset, name: str, dtype: Any) -> np.ndarray | None:
    return dataset[name].values.astype(dtype) if name in dataset else None


def read_unpacked(folder: str | Path) -> OmnicMap:
    """Rebuild the `OmnicMap` from an unpacked-map folder; it equals what `read_omnic` returned for the .map file.

    `path` is the original source path (the file need not exist). `collection.raw` is empty and `blocks` is an empty
    list, because the raw bytes are not kept.

    Raises:
        OmnicFormatError: if the folder is not an unpacked map of format version 1.
    """
    folder = Path(folder)
    meta = read_map_json(folder)
    md = meta["metadata"]
    dataset = _open_spectra(folder, meta)
    row, column = _optional(dataset, "grid_row", np.int64), _optional(dataset, "grid_column", np.int64)
    return OmnicMap(
        path=Path(meta["source"]["path"]),
        layout=md["layout"],
        axis=dataset[axis_dimension(md["axis_unit"])].values.astype(np.float64),
        axis_unit=md["axis_unit"],
        spectra=dataset["spectra"].values.astype(np.float32),
        x=dataset["x"].values.astype(np.float64),
        y=dataset["y"].values.astype(np.float64),
        z=_optional(dataset, "z", np.float64),
        map_kind=md["map_kind"],
        grid_shape=(md["grid_shape"][0], md["grid_shape"][1]) if md["grid_shape"] else None,
        grid_index=(row, column) if row is not None and column is not None else None,
        titles=[str(title) for title in dataset["title"].values],
        times_s=dataset["time_s"].values.astype(np.float64),
        data_kind=md["data_kind"],
        collection=_collection(md["collection"]),
        backgrounds=_backgrounds(dataset, meta["backgrounds"], md["axis_unit"]),
        background_index=_optional(dataset, "background_index", np.int64),
        images=_images(folder, meta["files"], meta["images"]),
        map_definition=_definition(md["map_definition"]),
        map_description=_description(md["map_description"]),
        history=md["history"],
        title=md["title"],
        file_time=dt.datetime.fromisoformat(md["file_time"]),
        blocks=[],
    )
