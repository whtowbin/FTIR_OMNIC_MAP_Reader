# Exports of an `OmnicMap`

All examples use `tests/data/GCB_1C_ol54_ultrafast map_75um aperture and step.map` (an 11 x 13 grid of 432-point
spectra) and `GCB_1C_GC52_CPX best points-50x50um_256avg.map` (6 points). Every export keeps NaN as stored.

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

Writes `to_xarray()` to a Zarr store. It needs the optional extra: `pip install "ftir-OMNIC-map-reader[zarr]"`, and
raises an `ImportError` that says so otherwise. The default mode refuses to overwrite an existing store; pass
`mode="w"` to replace it. Read it back with `xarray.open_zarr(path)`; the result equals `to_xarray()`.

## `to_images()`

Returns a list of `ImageArray(index, format, array, extent)`:

- `array` is `(height, width, 3)` uint8, upright: the first row is the largest stage Y and the first column the
  smallest stage X.
- `extent` is `(xmin, xmax, ymin, ymax)` in stage um, ready for `plt.imshow(array, extent=extent)`; `None` if the file
  has no rectangle for that image.
- `format` is `"BMP"` for a video image (878 x 686 px) and `"JPEG"` for a mosaic tile; the tiles of one map share a
  coordinate system and are placed by their extents.

The orientation was verified against OMNIC's screenshots (docs/file_format.md, section 8). A map without images gives
an empty list.
