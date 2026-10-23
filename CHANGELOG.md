# Changelog

## 1.1.0

### Added

- `ftir_map.read_omnic(path)` returns an `OmnicMap`. It reads the file's block directory instead of PyMca's fixed
  byte offsets, and handles both container layouts ("Spectral Data File" and "Spectral Exte File"). It reads grid maps,
  line scans, point-to-point maps, array-detector maps and interferogram maps; it raises `OmnicFormatError` on a file
  it cannot read.
- `OmnicMap` holds the spectra, spectral axis, stage X, Y and Z, map kind, spectrum titles and times, backgrounds,
  video images and mosaic tiles with their stage extents, the collection parameters and the map definition.
- Exports: `to_xarray()`, `to_numpy()`, `to_dict()`, `to_zarr(path)` and `to_images()`. See docs/exports.md.
- Optional extra for Zarr: `pip install "ftir-OMNIC-map-reader[zarr]"`.
- docs/file_format.md describes the file format, and says which fields are verified.
- Image orientation was checked against OMNIC's screenshots: the images are upright and need no flip.
- Tests, including OMNIC's own CSV exports as ground truth, and GitHub Actions workflows for testing and publishing.

### Changed

- `Load_Omnic_Map` is now built on `read_omnic`. Grid maps come back as before: spectra, wavenumbers and metadata are
  identical, except on array-detector maps, where the x and y positions now come from the stored float32 values instead
  of the rounded text in the spectrum titles (differences up to 0.003 um). Line scans, point-to-point maps and
  irregular maps, which used to fail, now come back per point on (point, wn). Unreadable files still give a warning
  and `None`.
- NaN values are kept as stored by `read_omnic` and its exports. `Load_Omnic_Map` still replaces them by 0, as before.
- The 7.1 MB array-detector sample map moved from `src/ftir_map/` to `tests/data/`. The wheel contains only package
  code, and the sdist does not contain `tests/data`.
- Pillow is now a direct dependency (it already came with matplotlib); it decodes the images.

### Deprecated

- `ftir_map.PyMca_OmnicMap`, `ftir_map.PyMca_Array_OmnicMap` and `ftir_map.LoadOmnicArrayMAP` emit a
  `DeprecationWarning` on import and will be removed in 2.0. Use `read_omnic`. Nothing in the package depends on them
  except these modules themselves.

## 1.0.0

First release: `Load_Omnic_Map` reading OMNIC .map files into xarray, based on PyMca's reader.
