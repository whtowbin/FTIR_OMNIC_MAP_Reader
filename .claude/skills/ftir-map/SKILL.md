---
name: ftir-map
description: Use when reading or exporting Thermo OMNIC .map files (FTIR microscope maps) with the ftir-OMNIC-map-reader package (`import ftir_map`) - grid, line, point-to-point, array-detector or interferogram maps; band-area images, line profiles, stage positions, video/mosaic images, exports to xarray/NumPy/Zarr/CSV, unpacked project folders, the `ftir-map` command line, or the legacy `Load_Omnic_Map`. Gives the API, the data types and shapes, and the traps.
---

# Working with OMNIC .map files through `ftir_map`

`ftir-OMNIC-map-reader` (PyPI; import `ftir_map`; Python 3.10+) reads Thermo OMNIC `.map` files without OMNIC. This
skill covers the package API, not spectroscopy (baselines, band assignment, thickness): decide those separately.
Full docs: https://ftir-omnic-map-reader.readthedocs.io/

```bash
pip install ftir-OMNIC-map-reader            # or: uv add ftir-OMNIC-map-reader
# extras: [yaml] metadata as YAML, [parquet] points tables as Parquet, [netcdf] spectra as .nc
```

## 1. Read a map

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")          # -> OmnicMap; raises OmnicFormatError if it cannot read it
m.map_kind                             # "grid", "line", "points" or "array"
m.spectra.shape                        # (n_points, n_axis), float32, ALWAYS in file (acquisition) order
m.axis                                 # (n_axis,) float64, ascending cm-1 (interferograms: point index as stored)
m.x, m.y                               # (n_points,) stage position of each spectrum in um
```

`OmnicFormatError` (a `ValueError`) means: not an OMNIC map, no spectra (a mosaic-only file), or a truncated or
inconsistent file. Catch it when looping over a folder of maps.

## 2. The `OmnicMap` fields

| Field | Type | Meaning |
|---|---|---|
| `spectra` | `ndarray (n_points, n_axis)` float32 | spectra in file order; NaN kept as stored |
| `axis` | `ndarray (n_axis,)` float64 | ascending cm-1 for spectra; for interferograms the point index in stored order (can run 8191 ... 0) |
| `wavenumber` | property | same as `axis` in cm-1; **raises `ValueError` for interferogram maps** |
| `axis_unit` | `str` | `"cm-1"`, or `"points"` for interferograms |
| `data_kind` | `str` | `"absorbance"`, `"transmittance"`, `"single_beam"`, `"interferogram"`; use this, not `collection.data_kind` (see Traps) |
| `x`, `y` | `ndarray (n_points,)` float64 | stored float32 stage positions, um; NaN if the file has none |
| `z` | `ndarray` or `None` | stage Z (focus), um; only classic-layout point maps record it |
| `map_kind` | `str` | `"grid"`, `"line"`, `"points"`, `"array"` |
| `grid_shape` | `(n_rows, n_columns)` or `None` | set only when every grid cell holds exactly one spectrum |
| `grid_index` | `(row, column)` int arrays or `None` | grid cell of each spectrum (grid and gridded array maps alike); row 0 = smallest Y, column 0 = smallest X |
| `n_points` | property, `int` | number of spectra |
| `titles` | `list[str]` | per-spectrum titles ("Spectrum 1 of 143", "Pixel 16, XPos=...") |
| `times_s` | `ndarray (n_points,)` | seconds since the first spectrum |
| `backgrounds` | `list[Background]` | `.spectrum`, `.wavenumber`, `.data_kind`, `.title`, `.time` |
| `background_index` | `ndarray` or `None` | which background each spectrum was ratioed against; can be present (all 0) when `backgrounds` is empty |
| `images` | `list[StageImage]` | video images (BMP) or mosaic tiles (JPEG): `.format`, `.width`, `.height`, `.pixel_size_um`, `.data` (encoded bytes); `.stage_extent` is `(xmin, xmax, ymin, ymax)` um, but `.extent` is the raw `(x0, y0, x1, y1)` rectangle: use `.stage_extent` or `to_images()` |
| `collection` | `CollectionParameters` | `n_points`, `first_x`, `last_x`, `x_units`, `n_sample_scans`, `n_background_scans`, `laser_frequency`, `data_spacing`, ... |
| `map_definition` | `MapDefinition` or `None` | set-up: `kind`, `x_range`, `y_range`, `x_step`, `y_step`, `n_x`, `n_y`, `title`, `file_name` |
| `map_description` | `MapDescription` or `None` | PyMca's description: `first_location`, `last_location`, `x_step`, `y_step`; not stage um on point maps (point indices) and can be left over from another map on line scans: use `x`, `y` for positions |
| `layout` | `str` | `"classic"` ("Spectral Data File") or `"extended"` ("Spectral Exte File") container |
| `title`, `history`, `file_time`, `path` | | file title, history text, `datetime`, source path |

Methods: `spectra_grid()` (grid maps only: `(n_rows, n_columns, n_axis)`, NaN in empty cells), `to_xarray()`,
`to_numpy()`, `to_dict()`, `to_zarr(path, mode="w-")`, `to_images()`, `save_spectra/save_points/save_metadata(path,
format=None)`, `save_images(folder)`, `export_all(folder, overwrite=False)`, `unpack(folder)`, `block_bytes(key)` (raw
blocks this reader does not decode).

## 3. Map kinds decide the shapes

| `map_kind` | `grid_shape` | `to_xarray()["spectra"].dims` | Use |
|---|---|---|---|
| `grid` | set | `("y", "x", "wavenumber")` | images of band areas |
| `array` | set if the pixels fill a grid, else `None` | as `grid`, or as `points` | images |
| `line` | `None` | `("point", "wavenumber")` | profiles along the line |
| `points` | `None` | `("point", "wavenumber")` | per-point spectra |

Always branch on `m.grid_shape is not None`, not on `map_kind == "grid"`, so array maps are handled too. For an
interferogram map (`data_kind == "interferogram"`) the spectral dimension is called `"axis"`, not `"wavenumber"`.

`to_xarray()` details:
- grid: coords `x` (n_columns), `y` (n_rows) are absolute stage positions (um); data vars `spectra`, `time_s`,
  `background_index`, `backgrounds`.
- per point: dimension `point`; coords `x`, `y`, `z` (if recorded), `title`, `time_s`, `background_index`.
- `ds.attrs` holds JSON-safe metadata: `map_kind`, `data_kind`, `axis_unit`, `grid_shape`, `collection`,
  `map_definition`, `reader_version`, ...

`to_numpy()` returns `(spectra, axes, positions, metadata)`: for a grid `spectra` is `(n_rows, n_columns, n_axis)` and
`axes` has `wavenumber`, `x`, `y`; otherwise `spectra` is `(n_points, n_axis)` and `axes` has only `wavenumber`
(called `axis` for interferogram maps, as in `to_xarray()`). `positions` has `x`, `y` (plus `grid_index` for grids,
`z` if recorded).

## 4. Recipes

Band-area image of a grid map (wavenumber is ascending, so `slice(low, high)`):

```python
import numpy as np

ds = m.to_xarray()
band = ds["spectra"].sel(wavenumber=slice(3000, 3700)).integrate("wavenumber")  # dims (y, x), absorbance * cm-1
band.plot(x="x", y="y")          # xarray puts larger Y up; with plt.imshow use origin="lower"
iy, ix = np.unravel_index(np.nanargmax(band.values), band.shape)   # NaN-safe (empty cells)
peak_xy = float(band.x[ix]), float(band.y[iy])
```

Profile along a line scan:

```python
import numpy as np

distance = np.hypot(m.x - m.x[0], m.y - m.y[0])        # um from the first spectrum
keep = (m.axis >= 3000) & (m.axis <= 3700)
area = np.trapezoid(m.spectra[:, keep], m.axis[keep], axis=1)   # NumPy < 2.0: np.trapz
```

Positions over the video image or mosaic (images and positions share stage um):

```python
import matplotlib.pyplot as plt

for img in m.to_images():                  # ImageArray(index, format, array (h, w, 3) uint8, extent)
    if img.extent is not None:
        plt.imshow(img.array, extent=img.extent)   # array is upright: first row = largest Y (origin="upper")
plt.scatter(m.x, m.y, s=8, c="r")
plt.gca().set_aspect("equal")
```

Grid spectra in file order and back: `cube = m.spectra_grid()`; `cube[m.grid_index]` is `m.spectra` again.

## 5. Writing files

```python
m.export_all("out_folder")                 # spectra.zarr, points.csv, metadata.json, images/ + README.txt
m.save_spectra("spectra.csv")              # format from the extension: .csv .npz .nc .zarr
m.save_points("points.csv")                # x, y, title, time, background per spectrum (.csv .json .parquet)
m.save_metadata("metadata.json")           # .json .yaml .csv
m.to_zarr("map.zarr")                      # read back: xarray.open_zarr("map.zarr")
```

- `export_all` raises `FileExistsError` for a non-empty folder unless `overwrite=True`; it checks optional packages
  for every chosen format before writing anything.
- `to_zarr` defaults to `mode="w-"` (never overwrites). `mode="w"` replaces a file or a Zarr store but refuses an
  ordinary folder that holds other files.
- CSV cells that a spreadsheet would run as formulas are written with a leading `'`.

## 6. Many maps: unpacked projects

```python
from ftir_map import unpack, open_project, load, read_unpacked

project = unpack(["folder_of_maps/"], "my_project")   # every .map under the folder -> my_project/maps/<name>/
project.names                         # maps that were unpacked
project.catalog                       # pandas DataFrame, one row per source file (status "skipped: ..." for failures)
m2 = project.load(project.names[0])   # OmnicMap equal to read_omnic of the source file
project = open_project("my_project")  # reopen later
obj = load(path)                      # .map file, unpacked map folder or project folder -> OmnicMap or Project
```

- `catalog` columns: `name`, `folder`, `source_path`, `sha256`, `map_kind`, `n_points`, `axis_min`, `axis_max`,
  `axis_unit`, `data_kind`, `x_min`, `x_max`, `y_min`, `y_max`, `n_images`, `file_time`, `status` (`"ok"` or
  `"skipped: <reason>"`). Names are the file stems made filesystem-safe (spaces to `_`, at most 80 characters).
- An unreadable file (for example a mosaic-only map) becomes a catalog row with status `"skipped: <reason>"`; it does
  not stop the run. Re-running `unpack` skips maps whose checksum is unchanged (`overwrite=True` redoes them).
- Each map folder holds `map.json`, `spectra.zarr` (or `spectra.nc`/CSV), `points.csv` and `images/`, readable
  without this package; `read_unpacked(folder)` rebuilds the `OmnicMap`.
- Opening a folder someone else shared is checked first: unsafe Zarr codecs raise `UnsafeZarrStore`
  (`ftir_map.zarr_safety`), netCDF files pointing at other files raise `UnsafeNetcdfFile` (`ftir_map.unpacked`).

## 7. Command line

```bash
ftir-map info my_map.map                          # kind, number of spectra, axis, data kind, images, scans
ftir-map export my_map.map --out my_map_export    # like export_all; --spectra/--points/--images, --overwrite
ftir-map unpack "maps/" --out my_project          # like unpack(); --spectra, --png, --copy-source, --overwrite
ftir-map catalog my_project                       # list the maps of a project
```

Errors print one line on stderr and exit with code 1.

## 8. Legacy `Load_Omnic_Map` (version 1.0 API)

```python
from ftir_map.LoadOmnicMAP import Load_Omnic_Map

ds = Load_Omnic_Map("my_map.map")   # xr.Dataset; format="Xarray-DataArray" or "Dictionary" for the others
```

Kept for old scripts: it warns and returns `None` for unreadable files (instead of raising), replaces NaN by 0, and
gives grid `x`/`y` relative to the first column/row (not absolute stage positions). Its dimensions are
`("y", "x", "wn")` for grids and `("point", "wn")` otherwise (`wn`, not `wavenumber`); the `"Dictionary"` format has
keys `data`, `x`, `y`, `wn`, `metadata` (plus `x_stage`, `y_stage` for non-grid maps). Prefer `read_omnic` in new code.

## 9. Traps

- `m.spectra` is never a grid. Use `spectra_grid()`, `to_xarray()` or `to_numpy()` for `(row, column, axis)`.
- Plotting a spectra grid with `plt.imshow`: row 0 is the smallest Y, so pass `origin="lower"` (and
  `extent=(x0, x1, y0, y1)` from the `x`/`y` coordinates). Embedded images are the opposite: upright, `origin="upper"`.
- NaN is kept everywhere except `Load_Omnic_Map`. Use `np.nansum`/`np.nanmax` or `.fillna()` when cells may be empty.
- Positions are float32 values as stored, held as float64: compare them with a tolerance, not `==` across maps.
- A warning "the file stores N stage positions for M spectra" means the positions were ignored (NaN) because their
  count did not match; the map is then read per point.
- `m.wavenumber` raises on interferogram maps; use `m.axis` and check `m.axis_unit`.
- `m.data_kind` comes from the spectra block and is what the spectra are. `m.collection.data_kind` is the file's main
  header, which can differ after a conversion in OMNIC (the array test map: main header "transmittance", spectra
  "absorbance").
- `grid_shape` can be `None` for an array map with missing pixels; do not assume an image.
- Mosaic-only files have no spectra: `read_omnic` raises `OmnicFormatError`.
