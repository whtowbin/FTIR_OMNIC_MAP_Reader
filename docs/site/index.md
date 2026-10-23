# FTIR OMNIC Map Reader

`ftir-OMNIC-map-reader` (import name `ftir_map`) reads the `.map` files that Thermo OMNIC writes for FTIR microscope
maps, without OMNIC. It handles grid (area) maps, line scans, point-to-point maps and array-detector maps, and it
returns the spectra, the spectral axis, the stage position of every spectrum, the backgrounds, the video or mosaic
images with their stage coordinates, and the collection metadata that could be decoded.

From there you can work with NumPy arrays or xarray datasets, write the data to open formats (Zarr, CSV, NPZ, JSON,
YAML, PNG or TIFF with world files), or unpack a whole folder of `.map` files into a project folder of open files that
needs neither OMNIC nor this package to read.

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")
print(m.map_kind, m.n_points, m.spectra.shape)  # grid 143 (143, 432)
ds = m.to_xarray()  # spectra on (y, x, wavenumber) for a grid map
m.export_all("my_map_export")  # spectra.zarr, points.csv, metadata.json and images/ in one folder
```

## Where to go next

- [Installation](installation.md) and the [Quick start](quickstart.md).
- [Map kinds](map_kinds.md): what a grid, line, points and array map return.
- [Metadata and images](metadata.md) and [Exports](exports.md): what is stored, and every export with example output.
- [Unpacked projects](unpacked.md) and the [command line](cli.md): `ftir-map unpack` and the other commands.
- [File format notes](file_format.md): the `.map` layout, with what is verified.
- [Python API](api.md), [Migrating from 1.0](migrating.md) and the [Changelog and credits](changelog.md).

## Credits

This reader is built on the OMNIC reader in [PyMca](https://github.com/silx-kit/pymca) (V.A. Sole, European
Synchrotron Radiation Facility, MIT license). PyMca's code was the starting point and the key to decoding the rest of
the file format; [How this reader was made](changelog.md#how-this-reader-was-made) tells the story.

The source is at [github.com/whtowbin/FTIR_OMNIC_MAP_Reader](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader). Report
problems in its [issue tracker](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/issues).
