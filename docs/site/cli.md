# Command line

Installing the package adds the `ftir-map` command, which has four subcommands. Errors print one line to stderr
(`ftir-map: <message>`) and exit with status 1.

```console
$ ftir-map --help
usage: ftir-map [-h] {info,export,unpack,catalog} ...

Inspect and export Thermo OMNIC .map files.

positional arguments:
  {info,export,unpack,catalog}
    info                print a summary of a map
    export              write spectra, points, metadata and images to a folder
    unpack              unpack .map files into an open project folder
    catalog             list the maps of a project folder

options:
  -h, --help            show this help message and exit
```

## `ftir-map info`

Prints the map kind, number of points, axis range, data kind, number of images and key collection parameters.

```console
$ ftir-map info --help
usage: ftir-map info [-h] file

positional arguments:
  file

options:
  -h, --help  show this help message and exit
```

For example:

```console
$ ftir-map info "GCB_1C_ol54_ultrafast map_75um aperture and step.map"
file: GCB_1C_ol54_ultrafast map_75um aperture and step.map
map kind: grid (13 rows x 11 columns)
points: 143
axis: 678.849 to 4003.67 cm-1, 432 values
data kind: absorbance
images: 6
collection:
  n_sample_scans: 1
  n_background_scans: 1
  data_spacing: 7.714192189086065
  laser_frequency: 15798.666015625
  mirror_velocity: 3.1647000312805176
```

## `ftir-map export`

Reads one map and writes a folder with `export_all`: the spectra, a points table, the metadata and the images, with a
`README.txt` naming each file. It refuses to write into a folder that is not empty unless `--overwrite` is given. Every error (an unreadable or truncated
map, a missing optional package, a folder that is not empty) is one line on stderr and exit status 1, with no traceback. See
[Exports](exports.md#export_allfolder-spectrazarr-metadatajson-pointscsv-imagespng-overwritefalse) for the folder.

```console
$ ftir-map export --help
usage: ftir-map export [-h] --out OUT [--spectra {csv,nc,npz,zarr}] [--metadata {csv,json,yaml}]
                       [--points {csv,json,parquet}] [--images {png,tiff,original,none}]
                       [--overwrite]
                       file

positional arguments:
  file

options:
  -h, --help            show this help message and exit
  --out OUT             output folder
  --spectra {csv,nc,npz,zarr}
  --metadata {csv,json,yaml}
  --points {csv,json,parquet}
  --images {png,tiff,original,none}
  --overwrite           write into a folder that is not empty
```

The defaults are `--spectra zarr`, `--metadata json`, `--points csv` and `--images png`. `--spectra nc` (netCDF) is
opt-in and needs the `[netcdf]` extra; `--metadata yaml` needs `[yaml]` and `--points parquet` needs `[parquet]`.

```console
ftir-map export my_map.map --out my_map_export --spectra npz --metadata yaml --images tiff
```

## `ftir-map unpack`

Unpacks `.map` files, and every `.map` file under the folders you give, into a project folder. See
[Unpacked projects](unpacked.md).

```console
$ ftir-map unpack --help
usage: ftir-map unpack [-h] --out PROJECT [--spectra {zarr,csv,nc}] [--png] [--copy-source]
                       [--overwrite]
                       SRC [SRC ...]

positional arguments:
  SRC                   .map files and folders searched for .map files

options:
  -h, --help            show this help message and exit
  --out PROJECT         project folder
  --spectra {zarr,csv,nc}
                        spectra file; nc needs the [netcdf] extra
  --png                 also write the images as PNG
  --copy-source         copy each original .map file into its folder
  --overwrite           unpack again maps that are already in the project
```

It prints one line per source file (`unpacked`, `unchanged` or `skipped`) and a summary.

## `ftir-map catalog`

Prints the catalog of a project folder as a table: name, map kind, number of points, axis range, data kind, number of
images and status.

```console
$ ftir-map catalog --help
usage: ftir-map catalog [-h] project

positional arguments:
  project

options:
  -h, --help  show this help message and exit
```

The help texts above were copied from the `--help` output of this version. Run `ftir-map <command> --help` for the
text of the version you have installed.
