# Exports

All examples use `tests/data/GCB_1C_ol54_ultrafast map_75um aperture and step.map` (an 11 x 13 grid of 432-point
spectra) and `GCB_1C_GC52_CPX best points-50x50um_256avg.map` (6 points). Every export keeps NaN as stored.

Everything on this page starts from an `OmnicMap`, and each method is also a plain function in `ftir_map.exports` or
`ftir_map.writers` that takes the map first.

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")
ds = m.to_xarray()  # xarray.Dataset, attrs hold the metadata
spectra, axes, positions, metadata = m.to_numpy()  # arrays and plain dicts
everything = m.to_dict()  # spectra, axes, positions, titles, times, backgrounds, images
m.to_zarr("my_map.zarr")  # refuses to overwrite by default
images = m.to_images()  # video images or mosaic tiles with stage extents
print(ds.sizes)  # Frozen({'y': 13, 'x': 11, 'wavenumber': 432, 'background': 1})
```

The files these write are covered under [Writing files](#writing-files) below.

The spectral dimension is named `wavenumber` for spectra (cm-1) and `axis` for interferogram maps (data points).

## `to_xarray()`

### Grid maps and array-detector maps

```
<xarray.Dataset>
Dimensions:           (y: 13, x: 11, wavenumber: 432, background: 1)
Coordinates:
  * y                 (y) float64      stage Y of each row, um
  * x                 (x) float64      stage X of each column, um
  * wavenumber        (wavenumber) float64 678.8 686.6 ... 4004
  * background        (background) int64 0
    background_title  (background) <U36
Data variables:
    spectra           (y, x, wavenumber) float32
    time_s            (y, x) float64    seconds since the first spectrum
    background_index  (y, x) int64      which background each spectrum was ratioed against
    backgrounds       (background, wavenumber) float32
Attributes: source_path, reader_version, title, file_time, layout, map_kind, data_kind, axis_unit, grid_shape,
            n_points, collection, map_definition, map_description, history
```

- `x` and `y` are absolute stage positions (the mean of the points in each column or row). `Load_Omnic_Map` returns them
  relative to the first column and row instead.
- A grid cell with no spectrum would be NaN in `spectra` and `time_s`, and -1 in `background_index`.

### Line, point and irregular maps

```
<xarray.Dataset>
Dimensions:           (point: 6, wavenumber: 1725, background: 1)
Coordinates:
  * point             (point) int64 0 1 2 3 4 5
    x                 (point) float64   stage X, um
    y                 (point) float64   stage Y, um
    z                 (point) float64   stage Z (focus), um; only when the file records it (point maps)
    title             (point) <U15      spectrum title, e.g. "Spectrum 1 of 6"
    time_s            (point) float64   seconds since the first spectrum
    background_index  (point) int64     only when the file records it
  * wavenumber        (wavenumber) float64
  * background        (background) int64 0
    background_title  (background) <U1
Data variables:
    spectra           (point, wavenumber) float32
    backgrounds       (background, wavenumber) float32
```

The `backgrounds` variable is left out when the file holds no backgrounds. Points are in file (acquisition) order.

### `attrs`

JSON-serializable values only (`json.dumps(ds.attrs)` works). Raw header bytes are not included; use
`OmnicMap.block_bytes`. Tuples are lists, datetimes are ISO strings and non-finite numbers are `None`.

| Key | Content |
|---|---|
| `source_path`, `reader_version` | file read and package version |
| `title`, `file_time`, `history` | file title, file time (ISO), history text |
| `layout` | `"classic"` or `"extended"` container |
| `map_kind`, `data_kind`, `axis_unit` | `"grid"`, `"line"`, `"points"`, `"array"`; `"absorbance"`, `"transmittance"`, `"single_beam"`, `"interferogram"`; `"cm-1"` or `"points"` |
| `grid_shape`, `n_points` | `[n_rows, n_columns]` or `None`; number of spectra |
| `collection` | `n_points`, `first_x`, `last_x`, `x_units`, `data_kind`, `n_scan_points`, `interferogram_peak_position`, `n_sample_scans`, `n_fft_points`, `n_background_scans`, `background_gain`, `laser_frequency`, `spectrum_set_identifier`, `mirror_velocity`, `data_spacing` |
| `map_definition` | `kind`, `x_range`, `y_range`, `step`, `x_step`, `y_step`, `n_x`, `n_y`, `title`, `file_name`, `directory`; or `None` |
| `map_description` | `first_location`, `last_location`, `x_step`, `y_step`; or `None` |

## `to_numpy()`

Returns `(spectra, axes, positions, metadata)`.

| Item | Grid maps | Other maps |
|---|---|---|
| `spectra` | `(n_rows, n_columns, n_axis)`, NaN in empty cells | `(n_points, n_axis)` |
| `axes` | `{"wavenumber": (n_axis,), "x": (n_columns,), "y": (n_rows,)}` | `{"wavenumber": (n_axis,)}` |
| `positions` | `{"x": (n_points,), "y": (n_points,), "grid_index": (n_points, 2)}` plus `"z"` if recorded | `{"x", "y"}` plus `"z"` if recorded |
| `metadata` | same dict as `ds.attrs` | same |

`grid_index` holds the (row, column) of each point, so `spectra[tuple(grid_index.T)]` is the spectra in file order. For
interferogram maps the `"wavenumber"` key is called `"axis"`.

```python
spectra, axes, positions, metadata = m.to_numpy()
spectra.shape  # (13, 11, 432)
{k: v.shape for k, v in axes.items()}  # {'wavenumber': (432,), 'x': (11,), 'y': (13,)}
```

## `to_dict()`

A plain dict of NumPy arrays, lists and dicts:

| Key | Content |
|---|---|
| `spectra`, `axes`, `positions`, `metadata` | as in `to_numpy()` |
| `titles` | list of the per-spectrum titles |
| `time_s` | `(n_points,)` seconds since the first spectrum |
| `background_index` | `(n_points,)` int, or `None` |
| `backgrounds` | `(n_backgrounds, n_axis)`, or `None` |
| `images` | list of dicts with `index`, `format`, `array`, `extent` (see `to_images()`) |

## `to_zarr(path, mode="w-")`

Writes `to_xarray()` to a Zarr store. Zarr is a core dependency. The default mode refuses to overwrite an existing store; pass
`mode="w"` to replace it. `mode="w"` replaces a file or an existing Zarr store (one with a `zarr.json`, `.zgroup` or
`.zarray`), but raises `FileExistsError` for an ordinary folder that holds other files, because Zarr would delete
everything in it. `save_spectra(..., "zarr")` behaves the same. Read it back with `xarray.open_zarr(path)`; the result equals `to_xarray()`.

## `to_images()`

Returns a list of `ImageArray(index, format, array, extent)`:

- `array` is `(height, width, 3)` uint8, upright: the first row is the largest stage Y and the first column the
  smallest stage X.
- `extent` is `(xmin, xmax, ymin, ymax)` in stage um, ready for `plt.imshow(array, extent=extent)`; `None` if the file
  has no rectangle for that image.
- `format` is `"BMP"` for a video image (878 x 686 px) and `"JPEG"` for a mosaic tile; the tiles of one map share a
  coordinate system and are placed by their extents.

The orientation was verified against OMNIC's screenshots ([File format notes](file_format.md#8-images-and-registration)). A map without images gives
an empty list.

## Writing files

The writers below take the same `OmnicMap` and write files for other programs. Examples use `tests/data/GCB_1C_ol54_ultrafast map_75um aperture and step.map` (13 x 11 grid, 143 spectra, 6 mosaic tiles). The optional formats need an extra, and raise an `ImportError` that names it:

| Extra | Needed for |
|---|---|
| `[zarr]` | none: zarr is a core dependency, the extra stays as an alias |
| `[yaml]` | `.yaml` and `.yml` metadata (PyYAML) |
| `[parquet]` | `.parquet` points (pyarrow) |
| `[netcdf]` | `.nc` spectra (h5netcdf and h5py); netCDF is opt-in and never the default |

The format comes from the file extension unless `format=` is given. Each method is also a function in `ftir_map.writers`.

```python
m.save_metadata("meta.json")  # or .yaml, .csv
m.save_points("points.csv")  # one row per spectrum: x, y, row, column, title, time, background
m.save_spectra("spectra.npz")  # or .csv, .zarr; .nc is opt-in and needs the [netcdf] extra
m.save_images("images", format="png")  # PNG, TIFF or the original files, plus world files and an index
m.export_all("my_map_export")  # all of the above in one folder with a README.txt
```

!!! note "CSV and spreadsheets"
    Titles and other text in a CSV come from the `.map` file, so they are untrusted input. Every CSV field that
    starts with `=`, `+`, `-`, `@`, `'`, a tab or a carriage return gets a leading `'`, so that Excel and similar programs
    show it as text instead of running it as a formula. Numbers are written unchanged. `read_unpacked` removes exactly
    one quote again when it reads an unpacked `points.csv` back, so a title such as `'=x` comes back unchanged.

### `save_metadata(path, format=None)`

Writes the dict that is in `to_xarray().attrs` (see "attrs" above), without the raw header bytes. JSON, YAML and CSV hold the same content.

- `.json`: nested, indented. Arrays are lists, datetimes are ISO strings, non-finite numbers are `null`.
- `.yaml` or `.yml`: the same nested content in block style, written with `yaml.safe_dump`.
- `.csv`: one `key,value` row per value, with dotted keys for nested items and `key[0]` for list items. Empty fields are `None`.

```json
{
  "source_path": "tests/data/GCB_1C_ol54_ultrafast map_75um aperture and step.map",
  "map_kind": "grid",
  "grid_shape": [13, 11],
  "collection": { "n_points": 432, "first_x": 678.8489379882812, "n_sample_scans": 1 }
}
```

```csv
key,value
grid_shape[0],13
grid_shape[1],11
collection.n_sample_scans,1
map_definition.x_range[0],1124.2786262686
```

### `save_points(path, format=None)`

One row per spectrum in file order. Columns: `index`, `x_um`, `y_um`, `z_um` (only when the file records it), `row` and `column` (grid and array maps), `title`, `time_s`, `background_index` (only when recorded). Non-finite numbers are empty in CSV and `null` in JSON.

- `.csv`:

```csv
index,x_um,y_um,row,column,title,time_s,background_index
0,1124.2786865234375,19500.248046875,0,0,Spectrum 1 of 143,0.0,0
1,1199.2786865234375,19500.248046875,0,1,Spectrum 2 of 143,0.1,0
```

- `.json`: a list of records, `[{"index": 0, "x_um": 1124.27, ...}, ...]`.
- `.parquet`: the same columns; needs pyarrow (pandas is not needed).

The `row` and `column` columns link a spectrum to its cell in the `(y, x)` grid of `to_xarray()`. They tie `spectra.csv` columns (named by `index`) to grid positions.

### `save_spectra(path, format=None)`

An existing file or Zarr store at `path` is replaced.

- `.csv`: the first column is the spectral axis with its unit in the header (`wavenumber_cm-1`, or `axis_points` for interferograms), then one column per spectrum named by its point index. Values have nine significant digits, which holds a float32 exactly. Use `points.csv` to look up the position of column `n`.

```csv
wavenumber_cm-1,0,1,2,3
678.849,0.0412345,0.0398765,...
```

- `.npz`: arrays in file order: `spectra` `(n_points, n_axis)`, `axis`, `x`, `y`, `z` (when recorded), `grid_index` `(n_points, 2)` (grid maps) and `metadata`, the JSON text of the metadata. Read it with `np.load(path)` and `json.loads(str(npz["metadata"]))`; no pickle is used.
- `.nc` (opt-in, needs the `[netcdf]` extra): `to_xarray()` as NetCDF4, read back with `xarray.open_dataset(path)`. NetCDF attributes cannot be nested, so the metadata is one attribute, `metadata_json`. The backend is h5netcdf: it installs from wheels with h5py and needs no system netCDF library, while netCDF4 would add a compiled library for no gain here.
- `.zarr`: `to_zarr(path, mode="w")`.

### `save_images(folder, format="png")`

Writes `image_01`, `image_02`, ... for each video image or mosaic tile, each with a world file, plus an index. `format` is:

| `format` | Image | World file |
|---|---|---|
| `"png"` | `image_01.png` | `.pgw` |
| `"tiff"` | `image_01.tif` | `.tfw` |
| `"original"` | `image_01.bmp` or `.jpg`, the embedded bytes unchanged | `.bpw` or `.jgw` |

An image is decoded only in the format its block declares (BMP or JPEG). An image that cannot be decoded is skipped with a warning and has no record. `"original"` does not decode at all when the file gave the image size. A map without images writes nothing and returns `[]`; otherwise the function returns the index records.

The world file has six lines, in stage um: pixel size in x, 0, 0, minus the pixel size in y, and x and y of the upper-left pixel centre (half a pixel inside the corner `xmin`, `ymax`). The images are upright ([File format notes](file_format.md#8-images-and-registration)), so this places them in ImageJ, QGIS and similar programs. The stage coordinate system has no projection. Images with no stage rectangle get no world file.

```
0.580865673161
0
0
-0.588921282799
989.569058325
19747.9535862
```

`images.json` (a list of records) and `images.csv` index the images:

```csv
file,index,source_format,width,height,xmin,xmax,ymin,ymax,pixel_size_x_um,pixel_size_y_um,world_file,orientation_verified
image_01.png,1,JPEG,878,686,989.27,1499.27,19344.25,19748.25,0.5809,0.5889,image_01.pgw,True
```

`orientation_verified` is `True` for every image, since the orientation was checked against OMNIC's screenshots.

### `export_all(folder, spectra="zarr", metadata="json", points="csv", images="png", overwrite=False)`

Writes one self-describing folder and returns its path:

```
out/
  README.txt        names and describes each file
  spectra.zarr      or spectra.nc, .npz, .csv
  points.csv        or points.parquet, .json
  metadata.json     or metadata.yaml, .csv
  images/           image_01.png, image_01.pgw, ..., images.json, images.csv (left out if the map has no images)
```

`images="none"` leaves the folder out. All formats, and the optional packages they need (PyYAML, pyarrow, h5netcdf, zarr), are checked before anything is written. It raises `FileExistsError` if `folder` exists and is not empty, unless `overwrite=True`; files already in the folder that it does not write are kept.

## Command line

`ftir-map export FILE --out DIR` calls `export_all`; see [Command line](cli.md).
