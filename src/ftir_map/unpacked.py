"""Unpack one map into an open folder, and read it back.

`unpack_map(m, folder)` (also `m.unpack(folder)`) writes `map.json`, `spectra.zarr`, `points.csv`, `images/` and a
`README.txt`. Nothing in the folder needs this package or OMNIC to read. `read_unpacked(folder)` rebuilds the
`OmnicMap` from those files. The layout is specified in docs/site/unpacked_format.md.

The spectra, axis, positions, titles, times, backgrounds, images and decoded metadata round-trip exactly. The raw
header bytes (`collection.raw`) and the block directory (`blocks`) are not kept, so `OmnicMap.block_bytes` needs the
original .map file.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
import xarray as xr

from .exports import (
    axis_dimension,
    axis_name,
    backgrounds_array,
    build_metadata,
    dataset_to_zarr,
    import_optional,
    json_safe,
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
    header_axis,
)
from .writers import save_images, save_points, save_spectra, write_json
from .zarr_safety import check_zarr_store

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
FORMULA_PREFIXES = ("'", "=", "+", "-", "@", "\t", "\r")  # writers.csv_cell puts a quote in front of these
# (variable, points.csv column, dtype) for the variables a map may not have
CSV_OPTIONAL_COLUMNS = (
    ("z", "z_um", np.float64),
    ("grid_row", "row", np.int64),
    ("grid_column", "column", np.int64),
    ("background_index", "background_index", np.int64),
)
NC_COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}
# what read_unpacked needs from map.json; check_map_json reports every one that is missing or of the wrong type
SECTIONS = (("source", dict), ("files", dict), ("metadata", dict), ("backgrounds", list), ("images", list))
SOURCE_KEYS = ("file_name", "path", "size_bytes", "sha256")
FILES_KEYS = ("spectra", "spectra_format", "points", "images")
METADATA_KEYS = (
    "layout",
    "map_kind",
    "data_kind",
    "axis_unit",
    "grid_shape",
    "n_points",
    "collection",
    "map_definition",
    "map_description",
    "history",
    "title",
    "file_time",
)
BACKGROUND_KEYS = ("title", "time", "data_kind", "wavenumber_first", "wavenumber_last")
IMAGE_KEYS = ("index", "format", "file", "width", "height", "extent")
POINT_VARIABLES = ("spectra", "x", "y", "title", "time_s")


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
    """Write the embedded images (and PNG copies if asked); return what read_unpacked needs to rebuild them.

    An image that cannot be decoded is still kept as the original bytes when the file gave its size; otherwise it is
    left out, with a warning from `save_images`.
    """
    written = save_images(m, folder / IMAGES_DIR, "original")
    if png and written:
        save_images(m, folder / IMAGES_DIR / "png", "png")
    files = {record["index"]: record["file"] for record in written}  # an image that could not be decoded has none
    return [
        {
            "index": s.index,
            "format": s.format,
            "file": files[s.index],
            "width": s.width,
            "height": s.height,
            "extent": s.extent,
        }
        for s in m.images
        if s.index in files
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
    lines += ["", "Positions are stage coordinates in um. Format details: docs/site/unpacked_format.md in the package."]
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
    """Write `m` as an unpacked-map folder (docs/site/unpacked_format.md) and return the content of its `map.json`.

    `map.json` is written last, so a folder without one is an unfinished write.

    Args:
        m: the map to write.
        folder: the folder to write.
        spectra: "zarr" (default, writes spectra.zarr), "csv", or "nc" (netCDF4, opt-in, needs the [netcdf] extra).
        png: also write PNG copies of the images in `images/png/`.
        copy_source: copy the original .map file into `source/`.
        overwrite: replace the unpacked files in a folder that is not empty.
        sha256: the checksum of the source file, if the caller has it already.

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
        "backgrounds": json_safe(_background_records(m)),
        "images": json_safe(images),
    }
    write_json(folder / MAP_JSON, content)
    return content


# ------------------------------------------------------------------ reading
def _missing(values: dict[str, Any], keys: tuple[str, ...], where: str) -> list[str]:
    return [f"{where} has no {key!r}" for key in keys if key not in values]


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _section_problems(content: dict[str, Any]) -> list[str]:
    """Missing or mistyped top-level sections, then the keys inside each section that is there."""
    problems = []
    for key, kind in SECTIONS:
        if key not in content:
            problems.append(f"no {key!r}")
        elif not isinstance(content[key], kind):
            problems.append(f"{key!r} is not a {kind.__name__}")
    if isinstance(content.get("source"), dict):
        problems += _missing(content["source"], SOURCE_KEYS, "'source'")
    if isinstance(content.get("files"), dict):
        problems += _missing(content["files"], FILES_KEYS, "'files'")
        if content["files"].get("spectra_format") not in SPECTRA_FORMATS:
            problems.append(f"'files' spectra_format is not one of {list(SPECTRA_FORMATS)}")
    for kind, keys in (("backgrounds", BACKGROUND_KEYS), ("images", IMAGE_KEYS)):
        for i, record in enumerate(content[kind] if isinstance(content.get(kind), list) else []):
            problems += (
                _missing(record, keys, f"{kind}[{i}]") if isinstance(record, dict) else [f"{kind}[{i}] is not a dict"]
            )
    return problems


def _metadata_problems(metadata: dict[str, Any]) -> list[str]:
    """Missing keys in `metadata`, and a `grid_shape` or `n_points` that cannot be right."""
    problems = _missing(metadata, METADATA_KEYS, "'metadata'")
    n_points, shape = metadata.get("n_points"), metadata.get("grid_shape")
    if "n_points" in metadata and not (_is_int(n_points) and n_points > 0):
        problems.append(f"n_points is {n_points!r}, not a positive integer")
    elif shape is not None and "grid_shape" in metadata:
        if not (isinstance(shape, list) and len(shape) == 2 and all(_is_int(n) and n > 0 for n in shape)):
            problems.append(f"grid_shape is {shape!r}, not null or two positive integers")
        elif shape[0] * shape[1] != n_points:
            problems.append(f"grid_shape {shape} holds {shape[0] * shape[1]} points but n_points is {n_points}")
    collection = metadata.get("collection")
    if "collection" in metadata and not isinstance(collection, dict):
        problems.append("'collection' is not a dict")
    elif isinstance(collection, dict):
        required = tuple(
            f.name for f in dataclasses.fields(CollectionParameters) if f.default_factory is dataclasses.MISSING
        )
        problems += _missing(collection, required, "'collection'")
    description = metadata.get("map_description")
    if isinstance(description, dict):
        problems += _missing(description, ("first_location", "last_location", "x_step", "y_step"), "'map_description'")
    return problems


def check_map_json(content: object) -> None:
    """Raise one `OmnicFormatError` that lists every problem in the parsed content of a `map.json`.

    Checks the format name and version, the sections and keys `read_unpacked` needs and that `n_points` and
    `grid_shape` agree. It does not look at the other files; `check_unpacked_data` compares them with the spectra file.
    """
    if not isinstance(content, dict):
        raise OmnicFormatError(f"the content is a {type(content).__name__}, not an object")
    problems = []
    if content.get("format") != FORMAT_NAME:
        problems.append(f"format is {content.get('format')!r}, not {FORMAT_NAME!r}")
    if content.get("format_version") != FORMAT_VERSION:
        problems.append(f"format version {content.get('format_version')}, this reader reads {FORMAT_VERSION}")
    problems += _section_problems(content)
    if isinstance(content.get("metadata"), dict):
        problems += _metadata_problems(content["metadata"])
    if problems:
        raise OmnicFormatError("; ".join(problems))


def check_unpacked_data(content: dict[str, Any], dataset: xr.Dataset) -> None:
    """Raise one `OmnicFormatError` for every way the spectra file disagrees with a map.json that passed `check_map_json`.

    Compares `n_points`, the grid cells (`grid_row`, `grid_column` against `grid_shape`) and the number of background
    records with what the spectra file holds.
    """
    problems = [f"the spectra file has no {name!r}" for name in POINT_VARIABLES if name not in dataset]
    n_points, shape = content["metadata"]["n_points"], content["metadata"]["grid_shape"]
    if "spectra" in dataset and dataset["spectra"].sizes.get("point") != n_points:
        problems.append(
            f"n_points is {n_points} but the spectra file holds {dataset['spectra'].sizes.get('point')} spectra"
        )
    for name in ("x", "y", "title", "time_s"):
        if name in dataset and dataset[name].size != n_points:
            problems.append(f"{name!r} has {dataset[name].size} values, n_points is {n_points}")
    if shape is not None:
        for name, limit in (("grid_row", shape[0]), ("grid_column", shape[1])):
            if name not in dataset:
                problems.append(f"grid_shape is {shape} but the spectra file has no {name!r}")
            elif dataset[name].size != n_points or (
                dataset[name].size and not 0 <= dataset[name].min() <= dataset[name].max() < limit
            ):
                problems.append(f"{name!r} does not fit grid_shape {shape} and n_points {n_points}")
    stored = dataset["backgrounds"].sizes["background"] if "backgrounds" in dataset else 0
    if len(content["backgrounds"]) != stored:
        problems.append(
            f"{len(content['backgrounds'])} background records but {stored} backgrounds in the spectra file"
        )
    if problems:
        raise OmnicFormatError("; ".join(problems))


def read_map_json(folder: Path) -> dict[str, Any]:
    """The parsed `map.json` of an unpacked-map folder, checked by `check_map_json`."""
    path = folder / MAP_JSON
    if not path.is_file():
        raise OmnicFormatError(f"{folder}: no {MAP_JSON}; this is not an unpacked map folder")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        check_map_json(content)
    except json.JSONDecodeError as error:
        raise OmnicFormatError(f"{path}: not valid JSON ({error})") from None
    except OmnicFormatError as error:
        raise OmnicFormatError(f"{path}: {error}") from None
    return content


def _nan(value: float | None) -> float:
    return float("nan") if value is None else value


def _unescape_cell(text: str) -> str:
    """Undo the one quote that `writers.csv_cell` puts before text that a spreadsheet would run as a formula (or that
    starts with a quote itself)."""
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
    if len(points["x"]) != table.shape[1] - 1:
        raise OmnicFormatError(
            f"{folder}: {meta['files']['points']} has {len(points['x'])} rows but {meta['files']['spectra']} "
            f"has {table.shape[1] - 1} spectra"
        )
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
    if kind == "nc":
        import_optional("h5netcdf", "netcdf", "reading spectra.nc")
        check_netcdf_file(path)  # a shared file must not pull in bytes from other files
        opened = xr.open_dataset(path, engine="h5netcdf", decode_times=False, decode_timedelta=False)
    else:
        check_zarr_store(path)  # a shared store must not name a pickle or other unsafe codec
        # consolidated=False: read exactly the per-array metadata that was checked
        opened = xr.open_dataset(path, engine="zarr", consolidated=False, decode_times=False, decode_timedelta=False)
    with opened as dataset:
        return dataset.load()


class UnsafeNetcdfFile(ValueError):
    """A netCDF (HDF5) file that reads bytes from outside itself: external storage, a virtual dataset or an external link."""


def check_netcdf_file(path: Path) -> None:
    """Raise `UnsafeNetcdfFile` if an HDF5 file at `path` points to data in another file.

    HDF5 can store a dataset in any local file (external storage, virtual datasets) or link to another file, so a
    shared spectra.nc could copy the bytes of a file such as a key or a password list into the spectra. Everything in
    the file is checked before xarray opens it. Soft and hard links inside the file are fine.
    """
    h5py = import_optional("h5py", "netcdf", "reading spectra.nc")
    with h5py.File(path, "r") as file:
        pending, seen = [file], {hash(file.id)}
        while pending:
            group = pending.pop()
            for name in group:
                if isinstance(group.get(name, getlink=True), h5py.ExternalLink):
                    raise UnsafeNetcdfFile(f"{path}: {group.name}/{name} is a link to another file")
                item = group.get(name)
                if isinstance(item, h5py.Dataset) and (item.is_virtual or item.external):
                    raise UnsafeNetcdfFile(f"{path}: dataset {item.name} reads from another file")
                if isinstance(item, h5py.Group) and hash(item.id) not in seen:
                    seen.add(hash(item.id))
                    pending.append(item)


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


def _background_axis(record: dict[str, Any], n: int, collection: CollectionParameters) -> np.ndarray:
    """The wavenumbers of a background, built as `read_omnic` builds them.

    A background with the map's own header ends (the usual case) gets the reader's axis for that header: `linspace`
    from `first_x` to `last_x`, reversed if that runs downward in cm-1. That is not equal to `linspace(last, first)` in
    the last bits, and the stored ends alone do not say which direction the header had. Any other background gets the
    ascending `linspace` between its two stored ends.
    """
    first, last = record["wavenumber_first"], record["wavenumber_last"]
    if sorted((first, last)) != sorted((collection.first_x, collection.last_x)):
        return np.linspace(first, last, n)
    axis = header_axis(dataclasses.replace(collection, n_points=n))
    return axis[::-1].copy() if collection.x_units == "cm-1" and axis[0] > axis[-1] else axis


def _backgrounds(
    dataset: xr.Dataset, records: list[dict[str, Any]], axis_unit: str, collection: CollectionParameters
) -> list[Background]:
    """Backgrounds in the order and direction they were read: the file holds them on the ascending spectra axis."""
    out = []
    for row, record in zip(dataset["backgrounds"].values if records else [], records, strict=True):
        first, last = record["wavenumber_first"], record["wavenumber_last"]
        descending = axis_unit == "cm-1" and first > last
        spectrum = np.ascontiguousarray(row[::-1] if descending else row, dtype=np.float32)
        time = dt.datetime.fromisoformat(record["time"]) if record["time"] else None
        wavenumber = _background_axis(record, len(row), collection)
        out.append(Background(spectrum, wavenumber, record["data_kind"], record["title"], time))
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
    check_unpacked_data(meta, dataset)
    collection = _collection(md["collection"])
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
        collection=collection,
        backgrounds=_backgrounds(dataset, meta["backgrounds"], md["axis_unit"], collection),
        background_index=_optional(dataset, "background_index", np.int64),
        images=_images(folder, meta["files"], meta["images"]),
        map_definition=_definition(md["map_definition"]),
        map_description=_description(md["map_description"]),
        history=md["history"],
        title=md["title"],
        file_time=dt.datetime.fromisoformat(md["file_time"]),
        blocks=[],
    )
