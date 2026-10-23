"""Check a Zarr store before opening it, when it may come from someone else.

Unpacked maps and projects are meant to be shared. Zarr metadata names the data type and the codecs used to decode
each array, and some of those (an object data type with a pickle codec, for example) run arbitrary code when the
array is read. `check_zarr_store` reads only the JSON metadata of every array and refuses anything outside what
this package writes: numeric and boolean arrays, fixed-width strings, UTF-8 variable-length strings, and the
standard compressors. Open the store afterwards with `consolidated=False`, so that the metadata that was checked is
the metadata that is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

V2_COMPRESSORS = frozenset({"blosc", "zstd", "gzip", "zlib", "lz4", "bz2", "lzma"})
V3_CODECS = frozenset({"bytes", "zstd", "blosc", "gzip", "crc32c", "transpose", "vlen-utf8", "sharding_indexed"})
V3_DATA_TYPES = frozenset(
    {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    | {"float16", "float32", "float64", "string"}
)
V3_NAMED_DATA_TYPES = frozenset({"fixed_length_utf32", "fixed_length_ascii"})
SAFE_NUMPY_KINDS = frozenset("biufUS")  # bool, int, uint, float, fixed-width unicode, fixed-width bytes


class UnsafeZarrStore(ValueError):
    """A Zarr store whose metadata asks for a data type or codec this package will not decode."""


def check_zarr_store(path: str | Path) -> None:
    """Raise `UnsafeZarrStore` unless every array in the store at `path` uses a safe data type and codecs.

    Symbolic links inside the store are refused too, so chunk reads cannot reach files outside it.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise UnsafeZarrStore(f"{root} is not a Zarr store folder")
    for item in root.rglob("*"):
        if item.is_symlink():
            raise UnsafeZarrStore(f"{item} is a symbolic link inside the Zarr store")
        if item.name == ".zarray":
            _check_v2_array(item, _read_json(item))
        elif item.name == "zarr.json":
            meta = _read_json(item)
            if meta.get("node_type") == "array":
                _check_v3_array(item, meta)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise UnsafeZarrStore(f"{path}: unreadable Zarr metadata ({error})") from error
    if not isinstance(content, dict):
        raise UnsafeZarrStore(f"{path}: Zarr metadata is not a JSON object")
    return content


def _check_v2_array(path: Path, meta: dict[str, Any]) -> None:
    """Zarr format 2: `dtype`, `compressor` and `filters` in .zarray."""
    filters = [f.get("id") if isinstance(f, dict) else f for f in meta.get("filters") or []]
    try:
        dtype = np.dtype(meta.get("dtype"))
    except TypeError as error:
        raise UnsafeZarrStore(f"{path}: data type {meta.get('dtype')!r} is not allowed") from error
    if dtype.kind == "O":
        if filters != ["vlen-utf8"]:  # object arrays are only allowed as UTF-8 strings, never pickled objects
            raise UnsafeZarrStore(f"{path}: object data type with filters {filters} is not allowed")
    elif dtype.kind not in SAFE_NUMPY_KINDS or dtype.fields is not None or filters:
        raise UnsafeZarrStore(f"{path}: data type {meta.get('dtype')!r} with filters {filters} is not allowed")
    compressor = meta.get("compressor")
    if compressor is not None and (not isinstance(compressor, dict) or compressor.get("id") not in V2_COMPRESSORS):
        raise UnsafeZarrStore(f"{path}: compressor {compressor!r} is not allowed")


def _check_v3_array(path: Path, meta: dict[str, Any]) -> None:
    """Zarr format 3: `data_type` and `codecs` in zarr.json (sharding codecs are checked recursively)."""
    data_type = meta.get("data_type")
    named = isinstance(data_type, dict) and data_type.get("name") in V3_NAMED_DATA_TYPES
    if not (named or (isinstance(data_type, str) and data_type in V3_DATA_TYPES)):
        raise UnsafeZarrStore(f"{path}: data type {data_type!r} is not allowed")
    _check_v3_codecs(path, meta.get("codecs") or [])


def _check_v3_codecs(path: Path, codecs: list[Any]) -> None:
    for codec in codecs:
        name = codec.get("name") if isinstance(codec, dict) else codec
        if name not in V3_CODECS:
            raise UnsafeZarrStore(f"{path}: codec {name!r} is not allowed")
        if name == "sharding_indexed":
            config = codec.get("configuration", {})
            _check_v3_codecs(path, config.get("codecs", []) + config.get("index_codecs", []))
