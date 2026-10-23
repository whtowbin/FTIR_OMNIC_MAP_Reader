# FTIR OMNIC Map Reader

[![PyPI](https://img.shields.io/pypi/v/ftir-OMNIC-map-reader)](https://pypi.org/project/ftir-OMNIC-map-reader/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://ftir-omnic-map-reader.readthedocs.io/)
[![Tests](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/actions/workflows/tests.yml/badge.svg)](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/actions/workflows/tests.yml)

Read Thermo OMNIC `.map` files (FTIR area maps, line scans, point-to-point maps and array-detector maps) into Python,
without OMNIC. You get the spectra, the spectral axis, the stage position of every spectrum, the background spectra,
the video or mosaic images with their stage coordinates, and the collection metadata. Export them to open formats, or
unpack whole folders of `.map` files into a documented project folder that anyone can read.

**Full documentation: <https://ftir-omnic-map-reader.readthedocs.io/>**. This README is a short overview; the
documentation has every option, the output formats, worked examples and the Python API reference.

## Install

```console
pip install ftir-OMNIC-map-reader
```

With [uv](https://docs.astral.sh/uv/): `uv add ftir-OMNIC-map-reader`. Python 3.10 or newer. Optional extras add YAML
metadata, Parquet tables and netCDF export; see
[Installation](https://ftir-omnic-map-reader.readthedocs.io/en/latest/installation/).

## Quick start

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")
m.map_kind  # "grid", "line", "points" or "array"
m.spectra.shape  # (n_points, n_axis), in file order
m.axis  # wavenumbers in cm-1
m.x, m.y  # stage position of each spectrum in um
ds = m.to_xarray()  # spectra on (y, x, wavenumber) for a grid, (point, wavenumber) otherwise
m.export_all("my_map_export/")  # spectra, metadata, points table and images in one folder
```

More in the [Quick start](https://ftir-omnic-map-reader.readthedocs.io/en/latest/quickstart/) and
[Map kinds](https://ftir-omnic-map-reader.readthedocs.io/en/latest/map_kinds/).

## Command line

```console
ftir-map info my_map.map                         # what is in the file
ftir-map export my_map.map --out my_map_export   # spectra (Zarr), metadata, points, images
ftir-map unpack "Bristol 2017/" --out my_project # many maps into an open project folder
ftir-map catalog my_project                      # list the maps in a project
```

See [Command line](https://ftir-omnic-map-reader.readthedocs.io/en/latest/cli/) for every option.

## What else is in the documentation

- [Metadata and images](https://ftir-omnic-map-reader.readthedocs.io/en/latest/metadata/): collection parameters,
  map definition, backgrounds, and how to overlay spectra on the stage images.
- [Exports](https://ftir-omnic-map-reader.readthedocs.io/en/latest/exports/): xarray, NumPy with dictionaries, Zarr,
  CSV, JSON and YAML metadata, points tables, and images with world files.
- [Unpacked projects](https://ftir-omnic-map-reader.readthedocs.io/en/latest/unpacked/) and the
  [format specification](https://ftir-omnic-map-reader.readthedocs.io/en/latest/unpacked_format/): turn proprietary
  `.map` files into open files and load them back with this package.
- [File format notes](https://ftir-omnic-map-reader.readthedocs.io/en/latest/file_format/): how `.map` files are laid
  out, and which fields are verified.
- [Migrating from 1.0](https://ftir-omnic-map-reader.readthedocs.io/en/latest/migrating/): `Load_Omnic_Map` still
  works; the PyMca-derived modules are deprecated.
- [Python API](https://ftir-omnic-map-reader.readthedocs.io/en/latest/api/).

## Credits and license

The format knowledge behind the first versions comes from the OMNIC reader in
[PyMca](https://github.com/silx-kit/pymca) (V.A. Sole, European Synchrotron Radiation Facility, MIT license); see
[NOTICE](NOTICE). The rest of the format was worked out by comparing about 80 maps with OMNIC's own exports. MIT
license, see [LICENSE.txt](LICENSE.txt). Changes are listed in [CHANGELOG.md](CHANGELOG.md). Questions and bug reports:
[GitHub issues](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/issues).
