# Migrating from 1.0

Version 1.0 had one function, `Load_Omnic_Map`, built on code derived from PyMca. It still works in 1.1 and later. New
code should use `read_omnic`, which keeps the data as stored and has the exports described in this documentation.

## `Load_Omnic_Map` is kept

```python
from ftir_map.LoadOmnicMAP import Load_Omnic_Map

ds = Load_Omnic_Map("my_map.map")  # xarray.Dataset on (y, x, wn)
d = Load_Omnic_Map("my_map.map", format="Dictionary")  # keys data, x, y, wn, metadata
da = Load_Omnic_Map("my_map.map", format="Xarray-DataArray")  # the spectra only
```

`Load_Omnic_Map` is built on `read_omnic` now. For grid maps it returns what 1.0 did: the spectra, wavenumbers and
metadata are identical, and `x` and `y` are relative to the first column and row. What differs:

- **More map kinds.** Line scans, point-to-point maps and irregular maps used to fail. They now come back per point on
  `(point, wn)`, with `x` and `y` relative to the first point and the absolute stage positions in `x_stage` and
  `y_stage`. `attrs["map_kind"]` says which kind it is.
- **Array-detector maps.** The x and y positions now come from the stored float32 values instead of the rounded text in
  the spectrum titles. They differ by up to 0.003 um.
- **NaN.** `Load_Omnic_Map` still replaces NaN by 0 (grid and per-point maps), as PyMca did. `read_omnic` and its
  exports keep NaN as stored.
- **Units.** The per-point result of an interferogram map names its units from the file: `unit_names["wn"]` is
  `points` and `unit_names["data"]` is `interferogram`, not `cm^-1` and `absorbance`.
- **Errors.** A file `Load_Omnic_Map` cannot read gives a warning and `None`, as in 1.0. `read_omnic` raises
  `OmnicFormatError` (a `ValueError`) instead.

## From `Load_Omnic_Map` to `read_omnic`

| 1.0 | Now |
|---|---|
| `Load_Omnic_Map(path)` | `read_omnic(path).to_xarray()` |
| `ds["spectra"]` on `(y, x, wn)` | `ds["spectra"]` on `(y, x, wavenumber)` |
| `x`, `y` relative to the first column and row | absolute stage um; subtract the first value to get the old coordinates |
| `Load_Omnic_Map(path, format="Dictionary")` | `read_omnic(path).to_dict()` (different keys; see [Exports](exports.md)) |
| NaN replaced by 0 | NaN kept; use `ds["spectra"].fillna(0)` for the old behavior |

## Deprecated modules

`ftir_map.PyMca_OmnicMap`, `ftir_map.PyMca_Array_OmnicMap` and `ftir_map.LoadOmnicArrayMAP` are the modified PyMca
readers that 1.0 used. They emit a `DeprecationWarning` on import and will be removed in 2.0. Nothing in the package
depends on them except these modules themselves. Use `read_omnic`. They are not in the [Python API](api.md) reference.

## Other changes since 1.0

The [Changelog](changelog.md) lists every change. The ones that affect existing code:

- Pillow is a direct dependency now (it decodes the images); it already came with matplotlib.
- The 7.1 MB array-detector sample map moved from `src/ftir_map/` to `tests/data/` in the repository. The wheel contains
  only package code.
- `zarr` and `pandas` are core dependencies (the unpacked projects use them). The `[zarr]` extra stays as an empty
  alias. netCDF stays opt-in through the `[netcdf]` extra.
