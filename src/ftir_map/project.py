"""A project folder of unpacked maps: `unpack` many .map files, `open_project` to load them back, `load` for any path.

The folder holds `maps/<name>/` (one unpacked map each, see unpacked.py), `catalog.csv` for a spreadsheet and
`catalog.json` for programs. Re-running `unpack` skips every map that is already unpacked from the same file content.
The layout is specified in docs/unpacked_format.md.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import struct
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .exports import reader_version
from .omnic import OmnicFormatError, OmnicMap, read_omnic
from .unpacked import MAP_JSON, read_unpacked, sha256_file, unpack_map
from .writers import write_csv, write_json

PROJECT_FORMAT = "ftir-map project"
PROJECT_VERSION = 1
CATALOG_JSON = "catalog.json"
CATALOG_CSV = "catalog.csv"
MAPS_DIR = "maps"
OK = "ok"
CATALOG_COLUMNS = (
    "name",
    "folder",
    "source_path",
    "sha256",
    "map_kind",
    "n_points",
    "axis_min",
    "axis_max",
    "axis_unit",
    "data_kind",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "n_images",
    "file_time",
    "status",
)
NAME_LENGTH = 80
UNREADABLE = (OSError, ValueError, struct.error)  # OmnicFormatError is a ValueError

Row = dict[str, Any]
Progress = Callable[[Row, str], None]


# ------------------------------------------------------------------ finding sources and naming maps
def find_sources(sources: Iterable[str | Path], exclude: Path | None = None) -> list[Path]:
    """The given files, plus every .map file under the given folders, sorted, without duplicates.

    Anything inside `exclude` (the project folder) is left out, so a copied source is never found again.
    """
    found: dict[Path, None] = {}
    for source in map(Path, sources):
        if source.is_dir():
            candidates = sorted(p for p in source.rglob("*") if p.suffix.lower() == ".map" and p.is_file())
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"{source} does not exist")
        for path in candidates:
            resolved = path.resolve()
            if exclude is None or not resolved.is_relative_to(exclude):
                found[resolved] = None
    return list(found)


def safe_name(path: Path) -> str:
    """File stem with every character outside letters, digits, dot, dash and underscore replaced by `_`."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")[:NAME_LENGTH]
    return name or "map"


def unique_name(path: Path, sha256: str, taken: set[str]) -> str:
    """`safe_name`, with the first 8 characters of the checksum appended (then a counter) if it is already taken."""
    name = safe_name(path)
    if name in taken:
        name = f"{name}_{sha256[:8]}"
    candidate, count = name, 2
    while candidate in taken:
        candidate, count = f"{name}_{count}", count + 1
    return candidate


# ------------------------------------------------------------------ catalog
def _finite_range(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (None, None)


def catalog_row(name: str, m: OmnicMap, sha256: str) -> Row:
    """One catalog row for a map that was unpacked into `maps/<name>`."""
    x_min, x_max = _finite_range(m.x)
    y_min, y_max = _finite_range(m.y)
    return {
        "name": name,
        "folder": f"{MAPS_DIR}/{name}",
        "source_path": str(m.path),
        "sha256": sha256,
        "map_kind": m.map_kind,
        "n_points": m.n_points,
        "axis_min": float(m.axis.min()),
        "axis_max": float(m.axis.max()),
        "axis_unit": m.axis_unit,
        "data_kind": m.data_kind,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "n_images": len(m.images),
        "file_time": m.file_time.isoformat(),
        "status": OK,
    }


def skipped_row(name: str, path: Path, sha256: str, reason: str) -> Row:
    row = dict.fromkeys(CATALOG_COLUMNS)
    row.update(name=name, source_path=str(path), sha256=sha256, status=f"skipped: {reason}")
    return row


def _write_catalog(project_dir: Path, rows: list[Row]) -> None:
    content = {
        "format": PROJECT_FORMAT,
        "format_version": PROJECT_VERSION,
        "reader_version": reader_version(),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "maps": rows,
    }
    write_json(project_dir / CATALOG_JSON, content)
    write_csv(project_dir / CATALOG_CSV, CATALOG_COLUMNS, ([row[c] for c in CATALOG_COLUMNS] for row in rows))


# ------------------------------------------------------------------ unpacking many maps
def _is_current(existing: Row | None, sha256: str, project_dir: Path) -> bool:
    """True if the catalog row was unpacked from this file content and its folder is complete."""
    return bool(
        existing
        and existing["sha256"] == sha256
        and existing["status"] == OK
        and (project_dir / existing["folder"] / MAP_JSON).is_file()
    )


def _unpack_one(
    path: Path, project_dir: Path, existing: Row | None, taken: set[str], overwrite: bool, **options: Any
) -> tuple[Row, str]:
    """(catalog row, action) for one source file; action is "unpacked", "unchanged" or "skipped".

    `options` are the `unpack_map` options `spectra`, `png` and `copy_source`.
    """
    sha256 = sha256_file(path)
    if existing and not overwrite and _is_current(existing, sha256, project_dir):
        return existing, "unchanged"
    name = existing["name"] if existing else unique_name(path, sha256, taken)
    try:
        m = read_omnic(path)
    except UNREADABLE as error:
        reason = str(error).removeprefix(f"{path.name}: ")
        return skipped_row(name, path, sha256, reason), "skipped"
    unpack_map(m, project_dir / MAPS_DIR / name, overwrite=True, sha256=sha256, **options)
    return catalog_row(name, m, sha256), "unpacked"


def unpack(
    sources: Iterable[str | Path],
    project_dir: str | Path,
    *,
    spectra: str = "nc",
    png: bool = False,
    copy_source: bool = False,
    overwrite: bool = False,
    progress: Progress | None = None,
) -> Project:
    """Unpack .map files (and every .map file under the given folders) into the project folder `project_dir`.

    Each readable map goes to `maps/<name>/` (see `unpack_map` for the options) and every file, readable or not, gets
    a row in `catalog.csv` and `catalog.json`. A file that cannot be read (for example a mosaic-only map) is listed
    with status "skipped: <reason>" and does not stop the run. A map whose file content (sha256) matches the catalog is
    left alone unless `overwrite` is True, even if other options changed. `progress(row, action)` is called after each
    file, with action "unpacked", "unchanged" or "skipped". Returns the opened project.

    Raises:
        FileNotFoundError: if a source does not exist.
        ValueError: for an unsupported `spectra` format.
    """
    project_dir = Path(project_dir)
    previous = _read_catalog(project_dir) if (project_dir / CATALOG_JSON).is_file() else []
    by_source = {row["source_path"]: row for row in previous}  # one row per source file, in first-seen order
    taken = {row["name"] for row in previous}
    for path in find_sources(sources, exclude=project_dir.resolve()):
        row, action = _unpack_one(
            path,
            project_dir,
            by_source.get(str(path)),
            taken,
            overwrite,
            spectra=spectra,
            png=png,
            copy_source=copy_source,
        )
        by_source[str(path)] = row
        taken.add(row["name"])
        if progress:
            progress(row, action)
    project_dir.mkdir(parents=True, exist_ok=True)
    _write_catalog(project_dir, list(by_source.values()))
    return Project(project_dir)


# ------------------------------------------------------------------ opening a project
def _read_catalog(project_dir: Path) -> list[Row]:
    path = project_dir / CATALOG_JSON
    if not path.is_file():
        raise OmnicFormatError(f"{project_dir}: no {CATALOG_JSON}; this is not a project folder")
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("format") != PROJECT_FORMAT or content.get("format_version") != PROJECT_VERSION:
        raise OmnicFormatError(f"{path}: not an {PROJECT_FORMAT!r} catalog of version {PROJECT_VERSION}")
    return content["maps"]


class Project:
    """A project folder of unpacked maps. `len(project)` and iteration cover the maps that were unpacked.

    `names` lists them, `load(name)` returns an `OmnicMap`, `to_xarray(name)` its dataset. `rows` is the catalog as a
    list of dicts (skipped files included) and `catalog` the same as a pandas DataFrame.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[Row] = _read_catalog(self.path)

    def __repr__(self) -> str:
        return f"Project({str(self.path)!r}, {len(self)} maps, {len(self.rows) - len(self)} skipped)"

    @property
    def catalog(self):
        """One row per source file as a pandas DataFrame (see docs/unpacked_format.md for the columns)."""
        import pandas

        return pandas.DataFrame.from_records(self.rows, columns=CATALOG_COLUMNS)

    @property
    def names(self) -> list[str]:
        return [row["name"] for row in self.rows if row["status"] == OK]

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def folder(self, name: str) -> Path:
        """The unpacked-map folder of `name`; KeyError if there is no such map, with the names that exist."""
        for row in self.rows:
            if row["name"] == name and row["status"] == OK:
                return self.path / row["folder"]
        raise KeyError(f"{name!r} is not an unpacked map in {self.path}; names: {self.names}")

    def load(self, name: str) -> OmnicMap:
        return read_unpacked(self.folder(name))

    def to_xarray(self, name: str):
        return self.load(name).to_xarray()


def open_project(project_dir: str | Path) -> Project:
    """Open a project folder written by `unpack`. Raises OmnicFormatError if it has no catalog."""
    return Project(project_dir)


def load(path: str | Path) -> OmnicMap | Project:
    """Open whatever `path` is: a .map file (`read_omnic`), an unpacked map folder (`read_unpacked`) or a project."""
    path = Path(path)
    if path.is_file():
        return read_omnic(path)
    if (path / MAP_JSON).is_file():
        return read_unpacked(path)
    if (path / CATALOG_JSON).is_file():
        return open_project(path)
    raise FileNotFoundError(f"{path} is not a .map file, an unpacked map folder or a project folder")
