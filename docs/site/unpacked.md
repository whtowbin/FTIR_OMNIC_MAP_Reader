# Unpacked projects

Proprietary `.map` files are hard to archive and share. `ftir-map unpack` turns them into a folder of open files:
spectra in Zarr, positions in CSV, all metadata in JSON and the embedded images with world files. Nothing in that
folder needs OMNIC or this package to read, and this package can load it back into an `OmnicMap`.

## Unpack from the command line

```console
ftir-map unpack "Bristol 2017/" other_map.map --out my_project
ftir-map catalog my_project
```

Give `unpack` files and folders. It finds `.map` files recursively, writes one folder per readable map, and lists every
file, readable or not, in the catalog. A file it cannot read (for example a mosaic-only map with no spectra) is listed
as skipped and does not stop the run. The options are `--spectra zarr|csv|nc`, `--png`, `--copy-source` and
`--overwrite`; see [Command line](cli.md). `--spectra nc` is an opt-in netCDF export and needs the `[netcdf]` extra;
Zarr is the default.

Unpacking the five maps in `tests/data` prints one line per file and a summary:

```console
$ ftir-map unpack tests/data --out my_project
unpacked  921005-6-plaig2-1p-array-1: 1333 points, array, 12 images
unpacked  GCB_1C_GC52_CPX_best_points-50x50um_256avg: 6 points, points, 2 images
unpacked  GCB_1C_ol54_ultrafast_map_75um_aperture_and_step: 143 points, grid, 6 images
unpacked  GCB_4_Region_2_grain2_profile_2_50x100um_mosaic_image_test: 21 points, line, 6 images
unpacked  test_file_GCB_1C_GC52_opx_profile1_50x50um_25um_step_overlapping_points__256_ave: 12 points, line, 1 images
5 unpacked, 0 unchanged, 0 skipped -> my_project
```

## The folder tree

```
my_project/
  catalog.csv       one row per source file, for a spreadsheet
  catalog.json      the same, for programs
  maps/
    <name>/
      map.json          format name and version, source file, all metadata, per-background and per-image records
      spectra.zarr/     spectra, axis, positions, titles, times, backgrounds
      points.csv        one row per spectrum
      images/           only if the map has images: the embedded files byte for byte, with world files
      source/           only with --copy-source: the original .map file
      README.txt        what each file is, and how to open spectra.zarr from Python, R and Julia
```

`<name>` is the file name without its extension, with every character other than letters, digits, `.`, `-` and `_`
replaced by `_`. `map.json` is written last, so a folder without it is an unfinished write. The layout is specified in
the [format specification](unpacked_format.md) (version 1).

`ftir-map catalog my_project` prints the catalog as a table:

```console
$ ftir-map catalog my_project
name                                                                              map_kind  n_points  axis                  data_kind      n_images  status
921005-6-plaig2-1p-array-1                                                        array     1333      713.563-3999.81 cm-1  absorbance     12        ok
GCB_1C_GC52_CPX_best_points-50x50um_256avg                                        points    6         674.992-3999.81 cm-1  absorbance     2         ok
GCB_1C_ol54_ultrafast_map_75um_aperture_and_step                                  grid      143       678.849-4003.67 cm-1  absorbance     6         ok
GCB_4_Region_2_grain2_profile_2_50x100um_mosaic_image_test                        line      21        674.992-3999.81 cm-1  absorbance     6         ok
test_file_GCB_1C_GC52_opx_profile1_50x50um_25um_step_overlapping_points__256_ave  line      12        0-8191 points         interferogram  1         ok
```

## Load an unpacked project in Python

```python
import ftir_map

project = ftir_map.unpack(["Bristol 2017"], "my_project")  # or ftir_map.open_project("my_project")
print(project)  # for example Project('my_project', 5 maps, 0 skipped)
print(project.catalog[["name", "map_kind", "n_points"]])  # pandas DataFrame, one row per source file
name = project.names[0]
m = project.load(name)  # an OmnicMap, rebuilt from the open files
ds = project.to_xarray(name)

same = ftir_map.load("my_project/maps/" + name)  # reads one unpacked map folder, a project, or a .map file
print(same.map_kind)
```

- `ftir_map.unpack(sources, project_dir, ...)` is what the command calls. Its keyword arguments are `spectra`, `png`,
  `copy_source`, `overwrite` and a `progress` callback. It returns the opened `Project`.
- `ftir_map.open_project(folder)` opens an existing project; `Project` has `catalog`, `names`, `rows`, `folder(name)`,
  `load(name)` and `to_xarray(name)`, and supports `len()`, iteration and `in`.
- `ftir_map.read_unpacked(folder)` rebuilds the `OmnicMap` from one unpacked map folder. The spectra, axis, positions,
  titles, times, backgrounds, images and decoded metadata come back exactly. The raw header bytes and the block
  directory are not kept, so `block_bytes()` needs the original `.map` file.
- `ftir_map.load(path)` opens whatever `path` is: a `.map` file, an unpacked map folder or a project folder.
- `m.unpack(folder)` unpacks one already-read map into a single map folder (the same as `unpack_map`).

## Re-running

`unpack` is idempotent. A source whose path and sha256 are in the catalog and whose folder holds a `map.json` is
skipped (`unchanged` in the output). A source whose content changed is unpacked again into the same folder.
`--overwrite` unpacks every map again, replacing only the files the unpacker writes. Options such as `--spectra` and
`--png` do not apply to a map that is skipped as unchanged; use `--overwrite` to rewrite maps in another format.
Unreadable files are tried again on every run. A map that fails while it is being written (a full disk, say) is
listed as `skipped: cannot unpack: <error>` like an unreadable one, and the catalog is always written, even if the run
is interrupted. A file given twice under different spellings (another case of a folder name, a symbolic link) is one
source. Two maps whose names differ only in case (`Sample.map` and `sample.map`) get different folders, because macOS
and Windows treat those names as the same folder; the second gets the first 8 characters of its checksum. A folder
already in `maps/` that the catalog does not list also counts as taken. Names that Windows reserves (`CON`, `NUL`,
`COM1` and so on) get an underscore after the first part. Files inside the project folder are never taken as sources.

## Security: shared folders

Unpacked folders and projects are meant to be shared, so everything read from `map.json` and `catalog.json` is treated
as untrusted:

- A file or folder name from either file that points outside its folder (for example `../../x`) is refused with a
  `ValueError`, both when reading and when clearing files before an overwrite.
- `unpack` never writes through a symbolic link: if `maps/` or a map folder in the project is a symbolic link, it
  stops with an error instead of writing or deleting outside the project.
- A symbolic link at `catalog.csv` or `catalog.json` is refused too (`ValueError`), so the catalog cannot be written
  to a file outside the project.
- When a map is overwritten, a symbolic link at one of the names the unpacker writes (such as `spectra.zarr` or
  `images`) is removed as a link and not followed. The files are then written fresh inside the map folder, so a
  shared folder cannot redirect them.
- A shared `spectra.zarr` is checked before it is opened (`ftir_map.zarr_safety.check_zarr_store`, called by
  `read_unpacked`). Zarr metadata names the data type and codecs used to decode each array, and some of them, such as an
  object data type with a pickle codec, run code when the array is read. The check reads only the JSON metadata and
  refuses object types other than UTF-8 strings, pickle and other unknown codecs, structured types, and symbolic links
  inside the store, raising `UnsafeZarrStore` (a `ValueError`). The store is then opened with `consolidated=False`, so
  the metadata that was checked is the metadata that is used. Reading a Zarr store from someone else with this package
  cannot run a pickle codec.
- A shared `spectra.nc` is checked with h5py before it is opened (`ftir_map.unpacked.check_netcdf_file`, called by
  `read_unpacked`). HDF5 can store a dataset in any local file (external storage, virtual datasets) or link to another
  file, which would copy the bytes of that file into the spectra. A file that does any of these raises
  `UnsafeNetcdfFile` (a `ValueError`). Soft and hard links inside the file are fine.
- CSV text that a spreadsheet would run as a formula is quoted when written; see [Exports](exports.md).
