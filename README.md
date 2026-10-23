# FTIR OMNIC Map Reader

Read Thermo OMNIC `.map` files (FTIR maps, line scans and point lists) into Python as NumPy arrays or xarray datasets.
It returns the spectra, the spectral axis, the stage position of every spectrum, the background spectra, the video or
mosaic images with their stage coordinates, and the collection metadata that could be decoded.

## Install

```console
pip install ftir-OMNIC-map-reader
pip install "ftir-OMNIC-map-reader[zarr]"   # only if you want to write Zarr stores
```

With [uv](https://docs.astral.sh/uv/): `uv add ftir-OMNIC-map-reader`. Python 3.10 or newer.

## Quick start

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")
m.map_kind  # "grid", "line", "points" or "array"
m.spectra.shape  # (n_points, n_axis), in file order; NaN is kept as stored
m.axis  # wavenumbers in cm-1 (m.axis_unit == "cm-1")
m.x, m.y  # stage position of each spectrum in um
ds = m.to_xarray()  # spectra on (y, x, wavenumber) for a grid, (point, wavenumber) otherwise
```

`read_omnic` raises `OmnicFormatError` (a `ValueError`) for a file that is not an OMNIC map or holds no spectra.

## Map kinds

| `map_kind` | What it is | xarray layout of `spectra` |
|---|---|---|
| `grid` | rectangular area map | (y, x, wavenumber) |
| `array` | array-detector map; its pixels fill a grid | (y, x, wavenumber) |
| `line` | profile along a line | (point, wavenumber) |
| `points` | point-to-point map or irregular positions | (point, wavenumber) |

Interferogram maps (`m.data_kind == "interferogram"`) have `axis_unit == "points"`, and the last dimension is named
`axis`. `m.wavenumber` is the same as `m.axis` for spectra and raises an error for interferograms.

## Exports

Each export is described, with example output, in [docs/exports.md](docs/exports.md).

```python
ds = m.to_xarray()  # xarray.Dataset, attrs hold the metadata
spectra, axes, positions, metadata = m.to_numpy()  # arrays and plain dicts
everything = m.to_dict()  # spectra, axes, positions, titles, times, backgrounds, images
m.to_zarr("my_map.zarr")  # needs the zarr extra; refuses to overwrite by default
images = m.to_images()  # video images or mosaic tiles with stage extents
```

Showing a video image with the map positions:

```python
import matplotlib.pyplot as plt

for image in m.to_images():
    plt.imshow(image.array, extent=image.extent)  # extent = (xmin, xmax, ymin, ymax) in um; no flip needed
plt.plot(m.x, m.y, "r.")
```

## Metadata

`m.collection` (number of points, first and last X, scans, laser frequency, mirror velocity and more),
`m.map_definition` (kind, ranges, step, title, original file name), `m.title`, `m.file_time`, `m.history`, the
per-spectrum `m.titles`, `m.times_s` and `m.background_index`, and `m.backgrounds` (spectrum, title and time of each
background). `m.block_bytes(key)` returns the raw bytes of any block this reader does not decode. The same metadata,
as JSON-serializable values, is in `ds.attrs` and in the last item of `to_numpy()`. Which fields are verified against
OMNIC's own exports, and which are inferred, is in [docs/file_format.md](docs/file_format.md).

## Older interface

`Load_Omnic_Map` from version 1.0 still works and returns the same output for grid maps. It also reads line scans and
point maps now. A file it cannot read gives a warning and `None`.

```python
from ftir_map.LoadOmnicMAP import Load_Omnic_Map

ds = Load_Omnic_Map("my_map.map")  # xarray.Dataset on (y, x, wn)
d = Load_Omnic_Map("my_map.map", format="Dictionary")  # keys data, x, y, wn, metadata
```

It replaces NaN by 0, as PyMca did; `read_omnic` does not. The PyMca-derived modules `PyMca_OmnicMap`,
`PyMca_Array_OmnicMap` and `LoadOmnicArrayMAP` are deprecated and will be removed in 2.0.

## Credits and license

The format knowledge behind the first versions comes from the OMNIC reader in
[PyMca](https://github.com/vasole/pymca) (V.A. Sole, European Synchrotron Radiation Facility, MIT license). See
[NOTICE](NOTICE). The rest of the format was worked out by comparing 81 files with OMNIC's own exports. MIT license,
see [LICENSE.txt](LICENSE.txt). Changes are listed in [CHANGELOG.md](CHANGELOG.md); release steps in
[docs/RELEASING.md](docs/RELEASING.md).
