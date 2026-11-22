# Changelog

## Unreleased

### Changed

- Python 3.11 or newer is required: Python 3.10 reached end of life in October 2026, and NumPy and zarr have dropped
  it. pip on Python 3.10 keeps installing 1.1.1. Unpacked maps written on Python 3.10 (Zarr format 2, with
  `.zmetadata`) are still read and checked.
- zarr 3.1.2 and xarray 2025.6.1 are the minimum versions (were zarr 2.18 and xarray 2024.6). zarr 3.1.1 and older
  skip writing a chunk that compares equal to the fill value 0, so a spectrum of zeros came back with +0.0 where it
  held -0.0; xarray before 2025.6 cannot write zarr 3.1 stores. The test suite passes at these minimum versions.

### Fixed

- `read_unpacked` gave back the wavenumbers of backgrounds of a map with a descending cm-1 axis off by up to 4.5e-13;
  they now come back bit for bit, as `unpacked.md` says.
- `read_unpacked` and `open_project` raise `OmnicFormatError`, with every problem found in one message, for a damaged
  `map.json` or `catalog.json` (not JSON, a missing key, an `n_points` or `backgrounds` that does not match the spectra
  file, a `points.csv` with a row missing) instead of a raw `KeyError`, `AttributeError`, `IndexError` or `ValueError`,
  or no error at all. `ftir-map catalog` and `ftir-map unpack` print one line for a damaged catalog, not a traceback.
- `check_zarr_store` refuses a `sharding_indexed` codec without its `codecs` or `index_codecs` list (both required by
  the Zarr v3 spec); it used to read a missing list as empty and accept the store.
- `file_format.md`: the file time (0x128) is not the last background time on the extended maps (1 s, 51 s and 4 min
  21 s later in the three test files), and the number of sample scans (+36) is "consistent", not "verified": the
  interferogram map named "256 averages" stores 8. The point list in block 0x82 now has its offsets.

### Tests

- The command-line tests run `info`, `catalog`, `unpack` and `export` on the real test maps in-process and check the
  printed numbers, and the parser's defaults and choices against `cli.md`. Tests for damaged `map.json` and
  `catalog.json` files, for the folder names of maps with the same file name, for the checksum and size in `map.json`,
  and for the absence of `png/` and `source/` in a default unpack.
- Property tests with Hypothesis (`tests/test_properties.py`; 50 examples per property by default, 200 with
  `HYPOTHESIS_PROFILE=ci`): writer to reader to unpacked folder is bit-exact, `grid_layout` on any acquisition order,
  `csv_cell` and world-file inverses, and Zarr equals `to_xarray`.
- The writer tests check nested and existing output folders, the NPZ `z` and `grid_index` values, the CSV and world-file
  number formats, every key of `images.json`, the `format` argument of the `OmnicMap` methods, and the content of the
  CLI and netCDF exports of real maps. Tolerances now follow the documented digits.
- The synthetic writer gives every spectral-header, map-definition and map-description field its own value, and the
  tests decode each one from its offset (sample and background scans were both 16). `omnic.py` has tests for unsigned
  fields above 2**31, files cut by one byte at each size bound, 256-byte text fields, image sizes against Pillow, and
  the grid and line thresholds. Real-file checks use independent sources: background times against their date titles,
  the interferogram times against its date titles, the point list in block 0x82, the map definition and file names.
  Tolerances now follow the stated digits (7 significant digits of OMNIC's CSV, float32 rounding, 2-decimal titles).
- Tests for `to_xarray` dimensions, `units` attributes and `background_title`, the backgrounds of descending axes,
  both Zarr formats in `mode="w"`, the keys and calibration of `Load_Omnic_Map`, and the refusals of `check_zarr_store`
  (complex and structured dtypes, filters, compressors, unreadable and non-object metadata).

## 1.1.1 (2026-10-28)

### Added

- `ftir_map.__version__`.
- A skill for AI coding agents, `.claude/skills/ftir-map/SKILL.md` (API, data types, map kinds, recipes, traps), and
  the documentation page "Using with AI agents".

### Changed

- README is a short overview that links to each page of the documentation site.
- Documentation and NOTICE credit PyMca fully: "How this reader was made" describes how the reader was built on
  PyMca's OmnicMap.py, whose constants were the key to the file format. PyMca links point to silx-kit/pymca.

### Fixed

- `read_omnic` raises `OmnicFormatError`, not `struct.error`, `IndexError` or a NumPy error, on truncated files, short
  header blocks, image blocks without a length field and headers with 0 points; `Load_Omnic_Map` warns and returns
  `None` for them, as in 1.0.
- A key-6 position or background-index array whose length differs from the number of spectra now gives a warning
  with both counts (it used to leave NaN positions silently). This includes maps of more than 65535 spectra, whose
  16-bit count wraps.
- Grids with a step finer than 0.5 um are recognized as grids, not lines: the row and column tolerance is at most a
  tenth of the largest coordinate gap.
- Backgrounds of a map stored with a descending cm-1 axis are reversed to ascending, like the spectra.
- `Load_Omnic_Map` on a per-point map replaces NaN by 0, like the grid path. On both paths the units follow the file
  (`points` and `interferogram` for an interferogram map, not `cm^-1` and `absorbance`); every real grid map tested
  is absorbance in cm-1, so their labels do not change.
- `unpack`: maps whose names differ only in case (`Sample.map`, `sample.map`) get different folders; before, one
  overwrote the other on macOS and Windows and a re-run said "unchanged". A folder already in `maps/` that the catalog
  does not list is never reused.
- `unpack`: a map that fails while it is written (for example an image that cannot be decoded) is a skipped row with
  the error, not an abort; an error from the system (disk full, permissions) still stops the run. `catalog.json` and
  `catalog.csv` are always written. `save_images` writes the original
  images without decoding them when the file gave their size, and skips an image it cannot decode with a warning.
- `to_zarr(mode="w")` and `save_spectra(..., "zarr")` refuse an ordinary folder that holds files instead of deleting
  its content.
- `unpack` refuses a symbolic link at `catalog.csv` or `catalog.json`.
- `read_unpacked` checks a `spectra.nc` with h5py and raises `UnsafeNetcdfFile` for HDF5 external storage, virtual
  datasets and external links.
- `ftir-map info` on a truncated file prints one line on stderr and exits 1; a malformed sharding codec in a Zarr
  store raises `UnsafeZarrStore`, not `AttributeError`.
- `export_all` checks the optional packages of every chosen format before it writes anything; the CLI error says
  `--overwrite`.
- A CSV title that starts with an apostrophe followed by a formula character (`'=x`) is read back unchanged: text
  starting with `'` is quoted too, and the reader removes exactly one quote.
- The same map given under another spelling of its path (case, symbolic link) is one source, not two.
- `images` and `backgrounds` in `map.json` pass through `json_safe` (NaN is `null`); map names that Windows reserves
  (`CON`, `NUL`, `COM1`, ...) get an underscore.
- map_kinds.md: an `array` map has no `grid_shape` when its pixels do not fill a grid.

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
