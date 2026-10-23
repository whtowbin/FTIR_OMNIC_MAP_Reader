# Changelog

## Unreleased

### Added

- `ftir_map.__version__`.

### Changed

- README is a short overview that links to each page of the documentation site.

## 1.1.0 (2026-10-22)

### Added

- `ftir_map.read_omnic(path)` returns an `OmnicMap`. It reads the file's block directory instead of PyMca's fixed
  byte offsets, and handles both container layouts ("Spectral Data File" and "Spectral Exte File"). It reads grid maps,
  line scans, point-to-point maps, array-detector maps and interferogram maps; it raises `OmnicFormatError` on a file
  it cannot read.
- `OmnicMap` holds the spectra, spectral axis, stage X, Y and Z, map kind, spectrum titles and times, backgrounds,
  video images and mosaic tiles with their stage extents, the collection parameters and the map definition.
- Exports: `to_xarray()`, `to_numpy()`, `to_dict()`, `to_zarr(path)` and `to_images()`. See docs/site/exports.md.
- File writers: `save_metadata()` (JSON, YAML or flattened CSV), `save_points()` (CSV, JSON or Parquet),
  `save_spectra()` (CSV, NPZ or Zarr; netCDF opt-in), `save_images()` (PNG, TIFF or the original BMP and JPEG files,
  each with a world file and an `images.json` / `images.csv` index) and `export_all()` (one self-describing folder with
  a README.txt). See docs/site/exports.md.
- The `ftir-map` command: `ftir-map info FILE`, `ftir-map export FILE --out DIR`, `ftir-map unpack` and
  `ftir-map catalog`.
- Unpack maps into an open, documented folder: `ftir-map unpack SRC [SRC ...] --out PROJECT` and
  `ftir_map.unpack(sources, project_dir)`. Each map becomes `maps/<name>/` with `map.json`, `spectra.zarr` (Zarr),
  `points.csv`, the embedded images byte for byte with world files, and a `README.txt`. `catalog.csv` and
  `catalog.json` list every source file; unreadable files (such as mosaic-only maps) are listed as skipped. Re-running
  skips maps that are already unpacked. Options: `--spectra zarr|csv|nc`, `--png`, `--copy-source`, `--overwrite`.
  The format is in docs/site/unpacked_format.md.
- `read_unpacked(folder)` rebuilds the `OmnicMap` from an unpacked map (spectra bit for bit; the raw header bytes and
  block directory are not kept). `open_project(folder)` returns a `Project` with `catalog`, `names`, `load(name)` and
  `to_xarray(name)`. `ftir_map.load(path)` opens a `.map` file, an unpacked map or a project. `OmnicMap.unpack(folder)`.
- `ftir-map catalog PROJECT` lists the maps of a project.
- Optional extras: `[yaml]` (PyYAML), `[parquet]` (pyarrow) and `[netcdf]` (h5netcdf and h5py, for opt-in netCDF).
- Documentation site (MkDocs) on Read the Docs, built from docs/site. docs/site/file_format.md describes the file
  format and says which fields are verified.
- Image orientation was checked against OMNIC's screenshots: the images are upright and need no flip.
- Tests, including OMNIC's own CSV exports as ground truth, and GitHub Actions workflows for tests, docs and publishing.

### Security

- Paths read from `map.json` and `catalog.json` are confined to their folder, catalog names are validated before a
  folder is reused, and overwrites never follow symbolic links, so a shared unpacked folder or project cannot make the
  package read or delete files outside it.
- A shared `spectra.zarr` is checked before it is opened (`ftir_map.zarr_safety`): object data types other than UTF-8
  strings, pickle and other unknown codecs, symbolic links and misspelled metadata files are refused.
- Embedded images are decoded only in the format the file declares (BMP or JPEG).
- CSV text that a spreadsheet would run as a formula (starting with `=`, `+`, `-`, `@`, tab or CR) gets a leading quote.

### Changed

- `zarr` is now a core dependency, because `spectra.zarr` is the default spectra file of an unpacked map. netCDF is an
  opt-in export (`--spectra nc`, `spectra="nc"`) that needs the `[netcdf]` extra (`h5netcdf`, `h5py`). The `[zarr]` extra
  stays as an empty alias, so `pip install "ftir-OMNIC-map-reader[zarr]"` still works. `pandas` is a core dependency.
- `writers.write_csv`, `write_json`, `exports.backgrounds_array`, `exports.axis_dimension` and
  `exports.dataset_to_zarr` are public names now (they were private or inline).
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
