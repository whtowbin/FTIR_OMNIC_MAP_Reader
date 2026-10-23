"""File writers for an `OmnicMap`: metadata, per-point table, spectra, images with world files, and a whole folder.

Each function takes the `OmnicMap` first; `OmnicMap` has a method of the same name that calls it. The files are
described, with examples, in docs/exports.md. The metadata written is always `exports.build_metadata`, so JSON, YAML,
CSV, netCDF and NPZ carry the same content as `to_xarray().attrs`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .exports import (
    ImageArray,
    axis_name,
    build_metadata,
    import_optional,
    json_safe,
    to_images,
    to_xarray,
    to_zarr,
)
from .omnic import StageImage

FORMAT_ALIASES = {"yml": "yaml", "tif": "tiff", "jpg": "jpeg"}
ORIENTATION_VERIFIED = True  # docs/file_format.md, section 8: decoded images are upright, stage Y up
ORIGINAL_SUFFIX = {"BMP": ".bmp", "JPEG": ".jpg"}
WORLD_SUFFIX = {"PNG": ".pgw", "TIFF": ".tfw", "BMP": ".bpw", "JPEG": ".jgw"}
IMAGE_FORMATS = ("png", "tiff", "original")
IMAGE_INDEX_COLUMNS = (
    "file",
    "index",
    "source_format",
    "width",
    "height",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "world_file",
    "orientation_verified",
)

Columns = dict[str, list[Any]]


# ------------------------------------------------------------------ shared helpers
def resolve_format(path: str | Path, format: str | None, supported: Iterable[str]) -> str:
    """Format name from `format`, or from the file extension; ValueError if it is not one of `supported`."""
    name = (format or Path(path).suffix.lstrip(".")).lower()
    name = FORMAT_ALIASES.get(name, name)
    options = sorted(supported)
    if name not in options:
        raise ValueError(
            f"cannot write {Path(path).name} as {name or 'a file without extension'!r}; use one of {options}"
        )
    return name


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    """CSV with a header row; None becomes an empty field."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(["" if value is None else value for value in row] for row in rows)


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_columns_csv(columns: Columns, path: Path) -> None:
    _write_csv(path, columns, zip(*columns.values(), strict=True))


# ------------------------------------------------------------------ metadata
def flatten_metadata(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested metadata to dotted keys; list items are `key[0]`, `key[1]`. Empty containers stay as leaves."""
    if isinstance(value, dict) and value:
        pairs = ((f"{prefix}.{key}" if prefix else key, item) for key, item in value.items())
    elif isinstance(value, list) and value:
        pairs = ((f"{prefix}[{i}]", item) for i, item in enumerate(value))
    else:
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in pairs:
        flat.update(flatten_metadata(item, key))
    return flat


def _metadata_json(metadata: dict[str, Any], path: Path) -> None:
    _write_json(path, metadata)


def _metadata_yaml(metadata: dict[str, Any], path: Path) -> None:
    yaml = import_optional("yaml", "yaml", "YAML output", package="PyYAML")
    text = yaml.safe_dump(metadata, sort_keys=False, default_flow_style=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def _metadata_csv(metadata: dict[str, Any], path: Path) -> None:
    _write_csv(path, ["key", "value"], flatten_metadata(metadata).items())


METADATA_WRITERS: Mapping[str, Callable[[dict[str, Any], Path], None]] = {
    "json": _metadata_json,
    "yaml": _metadata_yaml,
    "csv": _metadata_csv,
}


def save_metadata(m, path: str | Path, format: str | None = None) -> None:
    """Write `build_metadata(m)` (without the raw header bytes) as JSON, YAML or CSV.

    The format comes from the extension (.json, .yaml, .yml, .csv) unless `format` is given. CSV has one `key,value`
    row per leaf with dotted keys (`collection.n_sample_scans`) and `key[0]` for list items.

    Raises:
        ValueError: for an unsupported format.
        ImportError: for YAML without the optional PyYAML package.
    """
    name = resolve_format(path, format, METADATA_WRITERS)
    METADATA_WRITERS[name](build_metadata(m), Path(path))


# ------------------------------------------------------------------ per-point table
def points_table(m) -> Columns:
    """One list per column, one entry per spectrum in file order.

    Columns: index, x_um, y_um, z_um (when recorded), row and column (grid maps), title, time_s, background_index
    (when recorded). Non-finite numbers are None.
    """
    columns: Columns = {"index": list(range(m.n_points)), "x_um": m.x.tolist(), "y_um": m.y.tolist()}
    if m.z is not None:
        columns["z_um"] = m.z.tolist()
    if m.grid_index is not None:
        columns["row"], columns["column"] = (index.tolist() for index in m.grid_index)
    columns["title"] = list(m.titles)
    columns["time_s"] = m.times_s.tolist()
    if m.background_index is not None:
        columns["background_index"] = m.background_index.tolist()
    return json_safe(columns)


def _points_json(columns: Columns, path: Path) -> None:
    _write_json(path, [dict(zip(columns, row, strict=True)) for row in zip(*columns.values(), strict=True)])


def _points_parquet(columns: Columns, path: Path) -> None:
    pyarrow = import_optional("pyarrow", "parquet", "Parquet output", package="pyarrow")
    parquet = import_optional("pyarrow.parquet", "parquet", "Parquet output", package="pyarrow")
    parquet.write_table(pyarrow.table(columns), path)


POINTS_WRITERS: Mapping[str, Callable[[Columns, Path], None]] = {
    "csv": _write_columns_csv,
    "json": _points_json,
    "parquet": _points_parquet,
}


def save_points(m, path: str | Path, format: str | None = None) -> None:
    """Write the per-point table (see `points_table`) as CSV, JSON (records) or Parquet.

    Raises:
        ValueError: for an unsupported format.
        ImportError: for Parquet without the optional pyarrow package.
    """
    name = resolve_format(path, format, POINTS_WRITERS)
    POINTS_WRITERS[name](points_table(m), Path(path))


# ------------------------------------------------------------------ spectra
def _spectra_csv(m, path: Path) -> None:
    """First column the spectral axis (unit in the header), then one column per point named by its index."""
    header = [f"{axis_name(m)}_{m.axis_unit}", *map(str, range(m.n_points))]
    table = np.column_stack([m.axis, m.spectra.T])
    np.savetxt(path, table, fmt="%.9g", delimiter=",", header=",".join(header), comments="")


def _spectra_npz(m, path: Path) -> None:
    """Per-point arrays in file order: spectra, axis, x, y, (z), (grid_index) and the metadata as a JSON string."""
    arrays: dict[str, Any] = {"spectra": m.spectra, "axis": m.axis, "x": m.x, "y": m.y}
    if m.z is not None:
        arrays["z"] = m.z
    if m.grid_index is not None:
        arrays["grid_index"] = np.column_stack(m.grid_index)
    arrays["metadata"] = np.array(json.dumps(build_metadata(m)))
    with open(path, "wb") as file:  # a file object stops NumPy from appending ".npz" to the name
        np.savez_compressed(file, **arrays)


def _spectra_netcdf(m, path: Path) -> None:
    """NetCDF4 through h5netcdf; netCDF attributes cannot be nested, so the metadata is one JSON string attribute."""
    import_optional("h5netcdf", "netcdf", "netCDF output")
    dataset = to_xarray(m)
    dataset.attrs = {"metadata_json": json.dumps(dataset.attrs)}
    dataset.to_netcdf(path, engine="h5netcdf")


def _spectra_zarr(m, path: Path) -> None:
    to_zarr(m, path, mode="w")


SPECTRA_WRITERS: Mapping[str, Callable[[Any, Path], None]] = {
    "csv": _spectra_csv,
    "npz": _spectra_npz,
    "nc": _spectra_netcdf,
    "zarr": _spectra_zarr,
}


def save_spectra(m, path: str | Path, format: str | None = None) -> None:
    """Write the spectra as .csv, .npz, .nc (netCDF) or .zarr; an existing file or Zarr store is replaced.

    Raises:
        ValueError: for an unsupported format.
        ImportError: for netCDF or Zarr without their optional packages.
    """
    name = resolve_format(path, format, SPECTRA_WRITERS)
    SPECTRA_WRITERS[name](m, Path(path))


# ------------------------------------------------------------------ images
def world_file_text(extent: tuple[float, float, float, float], width: int, height: int) -> str:
    """Six-line world file: pixel size x, 0, 0, minus pixel size y, then x and y of the upper-left pixel centre.

    `extent` is (xmin, xmax, ymin, ymax) in stage um. The images are upright, so the upper-left pixel centre is half
    a pixel inside the corner (xmin, ymax).
    """
    xmin, xmax, ymin, ymax = extent
    pixel_x, pixel_y = (xmax - xmin) / width, (ymax - ymin) / height
    values = [pixel_x, 0.0, 0.0, -pixel_y, xmin + pixel_x / 2, ymax - pixel_y / 2]
    return "\n".join(format(value, ".12g") for value in values) + "\n"


def _image_target(image: ImageArray, folder: Path, format: str) -> tuple[Path, str]:
    """(file path, format of the saved file: "PNG", "TIFF", "BMP" or "JPEG")."""
    stem = folder / f"image_{image.index:02d}"
    if format == "original":
        return stem.with_suffix(ORIGINAL_SUFFIX[image.format]), image.format
    saved = format.upper()
    return stem.with_suffix(".png" if format == "png" else ".tif"), saved


def _write_image(stage_image: StageImage, image: ImageArray, path: Path, saved_format: str) -> None:
    if saved_format == image.format:  # "original": the embedded file, byte for byte
        path.write_bytes(stage_image.data)
    else:
        Image.fromarray(image.array).save(path, format=saved_format)


def _image_record(image: ImageArray, path: Path, world_file: Path | None) -> dict[str, Any]:
    height, width = image.array.shape[:2]
    record: dict[str, Any] = {"file": path.name, "index": image.index, "source_format": image.format}
    record.update(width=width, height=height)
    if image.extent is None:
        record.update(xmin=None, xmax=None, ymin=None, ymax=None, pixel_size_x_um=None, pixel_size_y_um=None)
    else:
        xmin, xmax, ymin, ymax = image.extent
        record.update(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        record.update(pixel_size_x_um=(xmax - xmin) / width, pixel_size_y_um=(ymax - ymin) / height)
    record.update(world_file=world_file.name if world_file else None, orientation_verified=ORIENTATION_VERIFIED)
    return json_safe(record)


def save_images(m, folder: str | Path, format: str = "png") -> list[dict[str, Any]]:
    """Write each video image or mosaic tile into `folder`, with a world file, `images.json` and `images.csv`.

    `format` is "png", "tiff", or "original" (the embedded BMP or JPEG bytes). A world file (.pgw, .tfw, .bpw or .jgw)
    sits next to every image that has a stage rectangle and places it in stage um. Images are decoded only in the
    format their block declares. A map without images writes nothing. Returns the index records.

    Raises:
        ValueError: for an unsupported format.
    """
    format = resolve_format(folder, format, IMAGE_FORMATS)
    images = to_images(m)
    if not images:
        return []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for stage_image, image in zip(m.images, images, strict=True):
        path, saved_format = _image_target(image, folder, format)
        _write_image(stage_image, image, path, saved_format)
        world_file = None
        if image.extent is not None:
            world_file = path.with_suffix(WORLD_SUFFIX[saved_format])
            world_file.write_text(world_file_text(image.extent, image.array.shape[1], image.array.shape[0]))
        records.append(_image_record(image, path, world_file))
    _write_json(folder / "images.json", records)
    _write_csv(
        folder / "images.csv", IMAGE_INDEX_COLUMNS, ([record[c] for c in IMAGE_INDEX_COLUMNS] for record in records)
    )
    return records


# ------------------------------------------------------------------ whole folder
def _readme_text(m, names: dict[str, str]) -> str:
    lines = [f"Export of {m.path.name} ({m.map_kind} map, {m.n_points} spectra) by ftir-OMNIC-map-reader.", ""]
    lines += [f"{name}: {description}" for name, description in names.items()]
    lines += ["", "Positions are stage coordinates in um. Format details: docs/exports.md in the package repository."]
    return "\n".join(lines) + "\n"


def export_all(
    m,
    folder: str | Path,
    spectra: str = "zarr",
    metadata: str = "json",
    points: str = "csv",
    images: str | None = "png",
    overwrite: bool = False,
) -> Path:
    """Write spectra, per-point table, metadata and images into `folder` with a README.txt naming each file.

    Files: `spectra.<ext>`, `points.<ext>`, `metadata.<ext>`, `images/` (with its index and world files; left out
    when the map has no images or `images` is "none" or None) and `README.txt`. All formats are checked before
    anything is written.

    Raises:
        FileExistsError: if `folder` exists and is not empty and `overwrite` is False.
        ValueError: for an unsupported format.
        ImportError: for a format whose optional package is missing.
    """
    spectra_format = resolve_format("spectra", spectra, SPECTRA_WRITERS)
    metadata_format = resolve_format("metadata", metadata, METADATA_WRITERS)
    points_format = resolve_format("points", points, POINTS_WRITERS)
    images_format = None if images in (None, "none") else resolve_format("images", images, IMAGE_FORMATS)
    folder = Path(folder)
    if folder.is_dir() and any(folder.iterdir()) and not overwrite:
        raise FileExistsError(f"{folder} is not empty; pass overwrite=True to write into it")
    folder.mkdir(parents=True, exist_ok=True)

    names = {
        f"spectra.{spectra_format}": f"{m.n_points} spectra on the {axis_name(m)} axis ({m.axis_unit})",
        f"points.{points_format}": "one row per spectrum: stage position, title, time, background index",
        f"metadata.{metadata_format}": "collection parameters, map definition and file information",
    }
    save_spectra(m, folder / f"spectra.{spectra_format}", spectra_format)
    save_points(m, folder / f"points.{points_format}", points_format)
    save_metadata(m, folder / f"metadata.{metadata_format}", metadata_format)
    if images_format and save_images(m, folder / "images", images_format):
        names["images/"] = "video images or mosaic tiles, world files for GIS and ImageJ, images.json and images.csv"
    (folder / "README.txt").write_text(_readme_text(m, names), encoding="utf-8")
    return folder
