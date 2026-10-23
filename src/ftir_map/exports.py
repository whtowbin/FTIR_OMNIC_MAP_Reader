"""Exports of an `OmnicMap`: xarray, NumPy, plain dict, Zarr and the stage-registered images.

Each function takes the `OmnicMap` as its first argument; `OmnicMap` has a method of the same name that calls it.
The structure of every export is described in docs/exports.md.

The exports keep NaN exactly as stored in the file. Only the compatibility layer `Load_Omnic_Map` replaces NaN by 0.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import io
import math
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from PIL import Image

if TYPE_CHECKING:
    from .omnic import OmnicMap

PACKAGE_NAME = "ftir-OMNIC-map-reader"
NOT_GRIDDED_FILL_INDEX = -1  # background_index of a grid cell that holds no spectrum


@dataclasses.dataclass
class ImageArray:
    """A decoded video image or mosaic tile.

    `array` is (height, width, 3) uint8 and upright: the first row is the largest stage Y and the first column the
    smallest stage X, so `imshow(array, extent=extent)` places it correctly. `extent` is (xmin, xmax, ymin, ymax) in
    stage um, or None when the file has no rectangle for this image.
    """

    index: int
    format: str  # "BMP" or "JPEG"
    array: np.ndarray
    extent: tuple[float, float, float, float] | None


# ------------------------------------------------------------------ metadata
def reader_version() -> str:
    """Version of this package, or "unknown" when it is run from a source tree that was never installed."""
    try:
        return importlib_metadata.version(PACKAGE_NAME)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def import_optional(module: str, extra: str, feature: str, package: str | None = None) -> ModuleType:
    """Import an optional dependency, or raise an ImportError that names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{feature} needs the optional {package or module} package: pip install '{PACKAGE_NAME}[{extra}]'"
        ) from error


def axis_dimension(axis_unit: str) -> str:
    """Name of the spectral dimension: "wavenumber" for cm-1 data, otherwise "axis" (interferogram points)."""
    return "wavenumber" if axis_unit == "cm-1" else "axis"


def axis_name(m: OmnicMap) -> str:
    """Name of the spectral dimension of a map; see `axis_dimension`."""
    return axis_dimension(m.axis_unit)


def json_safe(value: Any) -> Any:
    """Convert a value to what JSON can hold: dicts, lists, strings, numbers, booleans and None.

    Tuples and arrays become lists, datetimes and paths become strings, and non-finite floats become None.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def build_metadata(m: OmnicMap) -> dict[str, Any]:
    """JSON-serializable metadata: collection parameters, map and data kind, map definition, source and version.

    The raw header bytes are left out; `OmnicMap.block_bytes` returns them.
    """
    collection = {f.name: getattr(m.collection, f.name) for f in dataclasses.fields(m.collection) if f.name != "raw"}
    collection["data_spacing"] = m.collection.data_spacing
    metadata = {
        "source_path": m.path,
        "reader_version": reader_version(),
        "title": m.title,
        "file_time": m.file_time,
        "layout": m.layout,
        "map_kind": m.map_kind,
        "data_kind": m.data_kind,
        "axis_unit": m.axis_unit,
        "grid_shape": m.grid_shape,
        "n_points": m.n_points,
        "collection": collection,
        "map_definition": dataclasses.asdict(m.map_definition) if m.map_definition else None,
        "map_description": dataclasses.asdict(m.map_description) if m.map_description else None,
        "history": m.history,
    }
    return json_safe(metadata)


# ------------------------------------------------------------------ grid helpers
def grid_axis_values(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    """Mean stored position of each grid column (or row): `index` says which of the n columns a point belongs to."""
    return np.array([values[index == k].mean() for k in range(n)])


def _grid_axes(m: OmnicMap) -> tuple[np.ndarray, np.ndarray]:
    """(x, y): the stage position of every grid column and row in um."""
    (n_rows, n_columns), (row, col) = _require_grid(m)
    return grid_axis_values(m.x, col, n_columns), grid_axis_values(m.y, row, n_rows)


def _require_grid(m: OmnicMap):
    if m.grid_shape is None or m.grid_index is None:
        raise ValueError(f"{m.path.name}: a {m.map_kind} map has no regular grid")
    return m.grid_shape, m.grid_index


def _scatter_to_grid(m: OmnicMap, values: np.ndarray, fill) -> np.ndarray:
    """Per-point values as an (n_rows, n_columns) array; cells without a spectrum get `fill`."""
    shape, index = _require_grid(m)
    out = np.full(shape, fill, dtype=values.dtype)
    out[index] = values
    return out


def backgrounds_array(m: OmnicMap) -> np.ndarray | None:
    """Backgrounds as (n_backgrounds, n_axis) float32 on the spectra's axis, or None if there are none.

    Returns None as well when a background has a different length from the spectra, which the files read so far
    never do.
    """
    if not m.backgrounds or any(len(b.spectrum) != len(m.axis) for b in m.backgrounds):
        return None
    rows = []
    for background in m.backgrounds:
        descending = m.axis_unit == "cm-1" and background.wavenumber[0] > background.wavenumber[-1]
        rows.append(background.spectrum[::-1] if descending else background.spectrum)
    return np.vstack(rows).astype(np.float32)


# ------------------------------------------------------------------ xarray
def to_xarray(m: OmnicMap) -> xr.Dataset:
    """Dataset of the map. Grid maps: `spectra` on (y, x, wavenumber); all others: (point, wavenumber).

    The last dimension is called "axis" instead of "wavenumber" for interferogram maps.
    """
    dataset = _grid_dataset(m) if m.grid_shape is not None else _point_dataset(m)
    background_array = backgrounds_array(m)
    if background_array is not None:
        dataset["backgrounds"] = (("background", axis_name(m)), background_array)
        dataset = dataset.assign_coords(
            background=np.arange(len(m.backgrounds)),
            background_title=("background", [b.title for b in m.backgrounds]),
        )
    dataset.attrs = build_metadata(m)
    return dataset


def _axis_coord(m: OmnicMap) -> tuple[str, tuple[str, np.ndarray, dict[str, str]]]:
    return axis_name(m), (axis_name(m), m.axis, {"units": m.axis_unit})


def _grid_dataset(m: OmnicMap) -> xr.Dataset:
    x, y = _grid_axes(m)
    name, axis_coord = _axis_coord(m)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {"spectra": (("y", "x", name), m.spectra_grid())}
    data_vars["time_s"] = (("y", "x"), _scatter_to_grid(m, m.times_s, np.nan))
    if m.background_index is not None:
        data_vars["background_index"] = (("y", "x"), _scatter_to_grid(m, m.background_index, NOT_GRIDDED_FILL_INDEX))
    coords = {"x": ("x", x, {"units": "um"}), "y": ("y", y, {"units": "um"}), name: axis_coord}
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _point_dataset(m: OmnicMap) -> xr.Dataset:
    name, axis_coord = _axis_coord(m)
    coords: dict[str, Any] = {
        "point": np.arange(m.n_points),
        name: axis_coord,
        "x": ("point", m.x, {"units": "um"}),
        "y": ("point", m.y, {"units": "um"}),
    }
    if m.z is not None:
        coords["z"] = ("point", m.z, {"units": "um"})
    coords["title"] = ("point", np.array(m.titles, dtype=str))
    coords["time_s"] = ("point", m.times_s, {"units": "s"})
    if m.background_index is not None:
        coords["background_index"] = ("point", m.background_index)
    return xr.Dataset(data_vars={"spectra": (("point", name), m.spectra)}, coords=coords)


# ------------------------------------------------------------------ NumPy and dict
def to_numpy(m: OmnicMap) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """(spectra, axes, positions, metadata).

    - spectra: (n_rows, n_columns, n_axis) for grid maps, otherwise (n_points, n_axis).
    - axes: "wavenumber" (or "axis" for interferograms), plus "x" and "y" for grid maps.
    - positions: per-point "x", "y", "z" (when recorded) and, for grid maps, "grid_index" as (n_points, 2) row and
      column. Points are in file order.
    - metadata: the same dict as the attrs of `to_xarray`.
    """
    axes = {axis_name(m): m.axis}
    if m.grid_shape is not None:
        axes["x"], axes["y"] = _grid_axes(m)
        spectra = m.spectra_grid()
    else:
        spectra = m.spectra
    positions = {"x": m.x, "y": m.y}
    if m.z is not None:
        positions["z"] = m.z
    if m.grid_index is not None:
        positions["grid_index"] = np.column_stack(m.grid_index)
    return spectra, axes, positions, build_metadata(m)


def to_dict(m: OmnicMap) -> dict[str, Any]:
    """Everything in the map as plain Python and NumPy objects.

    Keys: spectra, axes, positions, metadata (as in `to_numpy`), titles, time_s, background_index (or None),
    backgrounds ((n_backgrounds, n_axis) or None) and images (a list of dicts with index, format, array, extent).
    """
    spectra, axes, positions, metadata = to_numpy(m)
    images = [dataclasses.asdict(image) for image in to_images(m)]
    return {
        "spectra": spectra,
        "axes": axes,
        "positions": positions,
        "metadata": metadata,
        "titles": list(m.titles),
        "time_s": m.times_s,
        "background_index": m.background_index,
        "backgrounds": backgrounds_array(m),
        "images": images,
    }


# ------------------------------------------------------------------ Zarr
def to_zarr(m: OmnicMap, path: str | Path, mode: Literal["w", "w-"] = "w-") -> None:
    """Write `to_xarray(m)` to a Zarr store at `path`. `mode="w-"` refuses to overwrite an existing store.

    Raises:
        ImportError: if the optional zarr package is not installed.
        FileExistsError: if `mode="w-"` and `path` exists.
    """
    if mode == "w-" and Path(path).exists():
        raise FileExistsError(f"{path} exists; pass mode='w' to replace it")
    dataset_to_zarr(to_xarray(m), path, mode)


def dataset_to_zarr(dataset: xr.Dataset, path: str | Path, mode: Literal["w", "w-"] = "w-") -> None:
    """Write any dataset to a Zarr store, without zarr 3's warning about consolidated metadata."""
    import_optional("zarr", "zarr", "to_zarr")
    with warnings.catch_warnings():
        # zarr 3 warns that consolidated metadata is outside its spec; xarray still reads it back fastest
        warnings.filterwarnings("ignore", message="Consolidated metadata")
        dataset.to_zarr(path, mode=mode)


# ------------------------------------------------------------------ images
def to_images(m: OmnicMap) -> list[ImageArray]:
    """Video images (BMP) or mosaic tiles (JPEG) as upright RGB arrays with their stage extents.

    Orientation was checked against OMNIC's own screenshots (docs/file_format.md, section 8): the decoded images need
    no flip, and the extent is (xmin, xmax, ymin, ymax) for `imshow(array, extent=extent)` with the default
    `origin="upper"`.
    """
    out = []
    for stage_image in m.images:
        # Decode only the format the block declares ("BMP" or "JPEG"); never let Pillow guess from file content.
        with Image.open(io.BytesIO(stage_image.data), formats=[stage_image.format]) as decoded:
            array = np.asarray(decoded.convert("RGB"))
        out.append(ImageArray(stage_image.index, stage_image.format, array, stage_image.stage_extent))
    return out
