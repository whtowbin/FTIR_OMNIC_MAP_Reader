# Installation

ftir-OMNIC-map-reader needs Python 3.10 or newer (3.10 to 3.14 are tested). It is a pure Python package.

```console
pip install ftir-OMNIC-map-reader
```

With [uv](https://docs.astral.sh/uv/), in a project: `uv add ftir-OMNIC-map-reader`. To try it without adding it to a
project: `uv run --with ftir-OMNIC-map-reader python`.

The core install pulls in NumPy, pandas, xarray, Zarr, Pillow and matplotlib. Zarr is a core dependency because it is
the default spectra format of `ftir-map export` and `ftir-map unpack`.

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `[yaml]` | PyYAML | writing metadata as `.yaml` or `.yml` |
| `[parquet]` | pyarrow | writing the points table as `.parquet` |
| `[netcdf]` | h5netcdf, h5py | writing spectra as netCDF (`.nc`, `--spectra nc`); opt-in |
| `[zarr]` | nothing | an alias kept for compatibility; Zarr is already installed |

```console
pip install "ftir-OMNIC-map-reader[yaml,parquet]"
pip install "ftir-OMNIC-map-reader[netcdf]"   # only if you want netCDF files
```

Zarr is the default spectra format everywhere. netCDF is written only when you ask for it, and a request for it
without the `[netcdf]` extra raises an `ImportError` that names the extra.

## Check the install

```console
ftir-map --help
python -c "import ftir_map; print(ftir_map.__file__)"
```
