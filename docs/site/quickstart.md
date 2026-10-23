# Quick start

The examples use `my_map.map` for your own file. Every example on this page was run against the grid map
`GCB_1C_ol54_ultrafast map_75um aperture and step.map` from the repository's `tests/data` folder, and the printed values
below are what it gives.

## Read a map

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")
print(m.map_kind, m.grid_shape, m.n_points)  # grid (13, 11) 143
print(m.spectra.shape)  # (143, 432): one row per spectrum, in file order
print(m.data_kind, m.axis_unit)  # absorbance cm-1
print(m.axis[:3])  # [678.84893799 686.56313018 694.27732237]
print(m.x[:3], m.y[:3])  # stage X and Y of the first three spectra, in um
```

`read_omnic` raises `OmnicFormatError` (a `ValueError`) for a file that is not an OMNIC map or holds no spectra.

What you get back, in short:

| Attribute | Content |
|---|---|
| `m.map_kind` | `"grid"`, `"line"`, `"points"` or `"array"` (see [Map kinds](map_kinds.md)) |
| `m.spectra` | `(n_points, n_axis)` float32 array in file (acquisition) order. NaN is kept as stored |
| `m.axis`, `m.axis_unit` | the spectral axis, in cm-1 for spectra and `"points"` for interferogram maps |
| `m.x`, `m.y` | stage position of each spectrum in um (`m.z` for point maps that record it, otherwise `None`) |
| `m.grid_shape`, `m.grid_index` | `(n_rows, n_columns)` and the (row, column) of every spectrum, for grid and array maps |
| `m.titles`, `m.times_s` | title and acquisition time (seconds since the first spectrum) of every spectrum |

## Plot one spectrum

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(m.wavenumber, m.spectra[0])
ax.invert_xaxis()  # infrared spectra are usually drawn from high to low wavenumber
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Absorbance")
plt.show()
```

`m.wavenumber` is the same array as `m.axis` for spectra. For an interferogram map it raises `ValueError`, because the
axis is then a point index; use `m.axis` there.

## Plot a map as an image

For a grid or array map, `to_xarray()` puts the spectra on `(y, x, wavenumber)`. This integrates the OH stretching
region (3000 to 3800 cm-1) as the sum over the points times the point spacing, and draws it with the stage coordinates:

```python
ds = m.to_xarray()
band = ds["spectra"].sel(wavenumber=slice(3000, 3800))
area = band.sum("wavenumber", skipna=True) * m.collection.data_spacing  # absorbance times cm-1

fig, ax = plt.subplots()
area.plot(ax=ax, cbar_kwargs={"label": "Integrated absorbance (cm$^{-1}$)"})
ax.set_xlabel("Stage X (um)")
ax.set_ylabel("Stage Y (um)")
ax.set_aspect("equal")
plt.show()
```

This is a quick look, not a quantitative baseline-corrected integration. For a line or points map the dataset has
`(point, wavenumber)` dimensions instead; plot against `m.x` and `m.y`, or `ds["point"]`.

## Export everything

```python
from pathlib import Path

folder = m.export_all("my_map_export")  # raises FileExistsError if the folder exists and is not empty
print(sorted(path.name for path in folder.iterdir()))  # ['README.txt', 'images', 'metadata.json', 'points.csv', 'spectra.zarr']
```

`export_all` writes the spectra as Zarr (the default), the per-spectrum table as CSV, the metadata as JSON and the
images as PNG with world files, plus a `README.txt` that names each file. [Exports](exports.md) lists the other formats
and every individual writer.

## From the command line

```console
ftir-map info my_map.map
ftir-map export my_map.map --out my_map_export
```

See [Command line](cli.md) for all options, and [Unpacked projects](unpacked.md) to convert a folder of maps at once.
