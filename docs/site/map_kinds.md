# Map kinds

`read_omnic` classifies every file into one of four map kinds. The kind is in `m.map_kind`. Spectra are always stored
in `m.spectra` as `(n_points, n_axis)`, in the order OMNIC acquired them; the kind says how the positions are arranged.

| `map_kind` | What it is | `grid_shape`, `grid_index` | `to_xarray()` dimensions of `spectra` |
|---|---|---|---|
| `grid` | rectangular area map | set | `(y, x, wavenumber)` |
| `array` | array-detector map | set when its pixels fill a complete grid, else `None` | `(y, x, wavenumber)` with a grid, else `(point, wavenumber)` |
| `line` | profile along a line | `None` | `(point, wavenumber)` |
| `points` | point-to-point map or irregular positions | `None` | `(point, wavenumber)` |

- `grid_shape` is `(n_rows, n_columns)`. `grid_index` is a pair of integer arrays `(row, column)`, one entry per
  spectrum, so `m.spectra_grid()` returns the spectra as `(n_rows, n_columns, n_axis)`. Row 0 has the smallest stage Y
  and column 0 the smallest stage X. A layout is only recognized when every cell of the grid holds exactly one
  spectrum; otherwise `grid_shape` and `grid_index` are `None` and the map is read per point. Two stage coordinates
  are the same row or column when they differ by less than 0.5 um and by less than a tenth of the largest gap between
  distinct coordinates, so grids with steps finer than 0.5 um are recognized.
- The kind comes from the spectrum titles, the stage positions and the map definition stored in the file. A file
  whose spectra are titled "Pixel ..." is `array`. A map whose definition says "points" is `points`. Otherwise
  positions that fill a regular grid give `grid`, positions on one straight line give `line`, and everything else is
  `points`.
- An `array` map whose pixels do not fill a complete grid has `grid_shape` and `grid_index` set to `None`, like
  `line` and `points` maps.
- For `line` and `points` maps there is no grid; use `m.x` and `m.y`. `m.z` holds the stage Z (focus) of each spectrum
  when the file records it, which the classic layout does for point maps.

For the shapes in practice, here are the five maps in the repository's test data:

| File | `map_kind` | `spectra.shape` | `grid_shape` | Images |
|---|---|---|---|---|
| `GCB_1C_ol54_ultrafast map_75um aperture and step.map` | `grid` | (143, 432) | (13, 11) | 6 mosaic tiles |
| `921005-6-plaig2-1p-array-1.map` | `array` | (1333, 853) | (43, 31) | 12 |
| `GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map` | `line` | (21, 1725) | `None` | 6 mosaic tiles |
| `GCB_1C_GC52_CPX best points-50x50um_256avg.map` | `points` | (6, 1725) | `None` | 2 video images |
| `test file GCB_1C_GC52_opx profile1_..._Ifg.map` | `line` (interferograms) | (12, 8192) | `None` | 1 |

## The two file layouts

OMNIC writes two container layouts that differ in how the block directory is stored. The classic layout (magic
`Spectral Data File`, 32-bit offsets, the same container `.spa` files use) holds one block per spectrum with a header in
front of it. The extended layout (magic `Spectral Exte File`, 64-bit offsets) holds all spectra of the map in one
series block. `read_omnic` reads the block directory of either one and returns the same `OmnicMap`; `m.layout` says
which it found (`"classic"` or `"extended"`). The details are in [File format notes](file_format.md).

## Interferogram maps

A map saved as interferograms rather than spectra has `m.data_kind == "interferogram"` and `m.axis_unit == "points"`.
The axis is then the index of the data point. In exports the spectral dimension is called `axis` instead of
`wavenumber`, and `m.wavenumber` raises `ValueError`.

```python
from ftir_map import read_omnic

ifg = read_omnic("my_interferogram_map.map")
print(ifg.data_kind, ifg.axis_unit, ifg.spectra.shape)  # interferogram points (12, 8192)
print(ifg.to_xarray()["spectra"].dims)  # ('point', 'axis')
```
