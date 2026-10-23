# The "ftir-map unpacked" format, version 1

`ftir-map unpack` turns a proprietary OMNIC `.map` file into a folder of open files. Nothing in the folder needs OMNIC or
this package: the spectra are in Zarr, the tables in CSV and the metadata in JSON. The images are the bytes
OMNIC embedded (BMP or JPEG). `ftir_map.read_unpacked` and `ftir_map.load` read the folder back into an `OmnicMap`.

## One map

```
my_map/
  map.json          format name and version, source file, all metadata, per-background and per-image records
  spectra.zarr/     spectra, axis, positions, titles, times, backgrounds (a Zarr store: a folder)
  points.csv        one row per spectrum
  images/           only if the map has images
    image_01.bmp      the embedded file, byte for byte (.jpg for mosaic tiles)
    image_01.bpw      world file (.jgw for JPEG), places the image in stage um
    images.csv        one row per image: size and stage rectangle
    images.json       the same as JSON
    png/              only with --png: image_01.png, image_01.pgw, images.csv, images.json
  source/my_map.map only with --copy-source: the original file
  README.txt        what each file is, and how to open spectra.zarr from Python, R and Julia
```

`map.json` is written last. A folder without it is an unfinished write.

### map.json

| Key | Content |
|---|---|
| `format`, `format_version` | `"ftir-map unpacked"` and `1`. A reader must refuse any other version. |
| `reader_version` | Version of ftir-OMNIC-map-reader that wrote the folder. |
| `unpacked_at` | UTC time, ISO 8601. |
| `source` | `file_name`, `path`, `size_bytes`, `sha256` of the original `.map`, and `copied_to` (path under the map folder, or null). |
| `files` | `spectra` (file name), `spectra_format` (`zarr`, `csv` or `nc`), `points`, `images` (folder name). |
| `metadata` | Everything `build_metadata` gives: `title`, `file_time`, `layout`, `map_kind`, `data_kind`, `axis_unit`, `grid_shape`, `n_points`, `collection` (the decoded header fields), `map_definition`, `map_description`, `history`. Same content as `to_xarray().attrs`. |
| `backgrounds` | One record per background, in order: `title`, `time`, `data_kind`, `wavenumber_first`, `wavenumber_last`. The spectra are in the spectra file. |
| `images` | One record per image: `index`, `format` (`BMP` or `JPEG`), `file`, `width`, `height`, `extent` (x0, y0, x1, y1 as stored, stage um, or null). |

Numbers that are NaN or infinite in the source are `null` in JSON.

### spectra.zarr

Every map is stored as points in file order, grid maps too. A grid map is not reshaped; `grid_row` and `grid_column` say
which cell each point belongs to. A variable the map did not record is absent.

| Variable | Dimensions | Type | Content |
|---|---|---|---|
| `spectra` | point, wavenumber | float32 | Spectra as stored. NaN and infinity are kept. |
| `wavenumber` | wavenumber | float64 | Spectral axis, ascending, attribute `units` = `cm-1`. |
| `axis` | axis | float64 | Replaces `wavenumber` for interferogram maps, `units` = `points`. The dimension of `spectra` is then `axis`. |
| `point` | point | int64 | 0, 1, 2 ... in file order. |
| `x`, `y` | point | float64 | Stage position, `units` = `um`. NaN if not recorded. |
| `z` | point | float64 | Stage Z in um. Classic-layout point maps only. |
| `grid_row`, `grid_column` | point | int64 | Grid cell of each point, from 0. Grid and array maps only. |
| `title` | point | string | Spectrum title. |
| `time_s` | point | float64 | Seconds since the first spectrum. |
| `background_index` | point | int64 | Which background the spectrum was ratioed against (row of `backgrounds`). |
| `backgrounds` | background, wavenumber | float32 | Background single-beam spectra on the same axis. Absent if the map has none. |

Global attributes: `format`, `format_version`, `source_file`. NaN means missing. The same variables are written to
`spectra.nc` with `--spectra nc` (see below).

Opening without this package:

```python
import xarray as xr

ds = xr.open_zarr("spectra.zarr")  # or zarr.open("spectra.zarr")["spectra"][:]
```
```r
# Rarr package
spectra <- Rarr::read_zarr_array("spectra.zarr/spectra")
```
```julia
using Zarr; z = zopen("spectra.zarr"); spectra = z["spectra"][:, :]
```

MATLAB and Excel do not read Zarr; write `--spectra csv` for them.

### points.csv

Columns, in this order, as written by `save_points`: `index`, `x_um`, `y_um`, `z_um` (if recorded), `row`, `column`
(grid maps), `title`, `time_s`, `background_index` (if recorded). An empty cell is a missing number. A title that begins
with `=`, `+`, `-` or `@` has a leading `'` so that a spreadsheet does not run it as a formula.

### images

World files place each image in stage um (pixel size, rotation terms, then the upper-left pixel centre). Decoded images
are upright with stage Y increasing upward; see [file_format.md](file_format.md), section 8. `images.csv` has the columns
`file`, `index`, `source_format`, `width`, `height`, `xmin`, `xmax`, `ymin`, `ymax`, `pixel_size_x_um`,
`pixel_size_y_um`, `world_file`, `orientation_verified`.

### Other spectra formats

`--spectra nc` is an opt-in export: it writes `spectra.nc` (netCDF4 through h5netcdf, with the same variables as
`spectra.zarr`; gzip level 4, shuffled, no `_FillValue`) and needs `pip install "ftir-OMNIC-map-reader[netcdf]"`. Open it
with `xarray.open_dataset("spectra.nc", engine="h5netcdf")` or MATLAB's `ncread('spectra.nc', 'spectra')`.
`--spectra csv` writes `spectra.csv` (first column the axis, then one column per spectrum named by its index) and
`backgrounds.csv` in the same layout. CSV keeps 9 significant digits: the float32 spectra are exact, the axis is
rounded. Positions, titles and times come from `points.csv`.

### What is not kept

Reading an unpacked map gives the same `OmnicMap` as reading the `.map`, except for the raw bytes: `collection.raw` is
empty and `blocks` is an empty list, so `block_bytes()` needs the original file. `path` is the source path recorded in
`map.json`; the file need not exist. Backgrounds whose length differs from the spectra (not seen in any test file) are
not stored, with a warning.

## A project of many maps

```
project/
  catalog.csv       one row per source file, for a spreadsheet
  catalog.json      the same, for programs
  maps/
    <name>/           one map folder as above
```

`<name>` is the file name without extension, with every character other than letters, digits, `.`, `-` and `_` replaced
by `_`. If the name is taken by another file, the first 8 characters of the file's sha256 are appended.

### catalog.json and catalog.csv

`catalog.json` holds `format` (`"ftir-map project"`), `format_version`, `reader_version`, `updated_at` and `maps`, a list
of rows. `catalog.csv` has the same rows with these columns:

| Column | Content |
|---|---|
| `name`, `folder` | Name, and folder relative to the project (`maps/<name>`; empty if skipped). |
| `source_path`, `sha256` | Original file (resolved path) and its checksum. |
| `map_kind`, `data_kind` | `grid`, `array`, `line` or `points`; `absorbance`, `interferogram` and so on. |
| `n_points` | Number of spectra. |
| `axis_min`, `axis_max`, `axis_unit` | Range of the spectral axis and its unit. |
| `x_min`, `x_max`, `y_min`, `y_max` | Stage extent of the spectra in um. |
| `n_images` | Embedded images. |
| `file_time` | Time stored in the .map file. |
| `status` | `ok`, or `skipped: <reason>` for a file that cannot be read (for example a mosaic-only map). |

### Re-running

`unpack` is idempotent. A source whose path and sha256 are in the catalog and whose folder holds a `map.json` is skipped;
a source whose content changed is unpacked again into the same folder. `overwrite=True` (`--overwrite`) unpacks every
map again, replacing only the files the unpacker writes. Other options (`--spectra`, `--png`) do not apply to a map that
is skipped. Unreadable files are tried again every run. Files inside the project folder are never taken as sources.
