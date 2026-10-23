# Python API

The reference below is generated from the docstrings. The names in `ftir_map` itself are `read_omnic`, `OmnicMap`,
`OmnicFormatError`, `ImageArray`, `Project`, `load`, `open_project`, `unpack`, `read_unpacked` and `unpack_map`. The
methods of `OmnicMap` that export or write call the functions in `ftir_map.exports` and `ftir_map.writers`, which take
the map as their first argument.

The deprecated modules `PyMca_OmnicMap`, `PyMca_Array_OmnicMap` and `LoadOmnicArrayMAP` are not documented here; see
[Migrating from 1.0](migrating.md).

## Reading: `ftir_map.omnic`

::: ftir_map.omnic
    options:
      members:
        - read_omnic
        - OmnicMap
        - OmnicFormatError
        - CollectionParameters
        - Background
        - StageImage
        - MapDefinition
        - MapDescription
        - Block

## Exports: `ftir_map.exports`

::: ftir_map.exports
    options:
      members:
        - ImageArray
        - to_xarray
        - to_numpy
        - to_dict
        - to_zarr
        - to_images
        - build_metadata
        - backgrounds_array
        - axis_dimension
        - dataset_to_zarr

## Writers: `ftir_map.writers`

::: ftir_map.writers
    options:
      members:
        - save_metadata
        - save_points
        - save_spectra
        - save_images
        - export_all
        - points_table
        - write_csv
        - write_json

## Unpacked maps: `ftir_map.unpacked`

::: ftir_map.unpacked
    options:
      members:
        - unpack_map
        - read_unpacked

## Zarr store check: `ftir_map.zarr_safety`

::: ftir_map.zarr_safety
    options:
      members:
        - check_zarr_store
        - UnsafeZarrStore

## Projects: `ftir_map.project`

::: ftir_map.project
    options:
      members:
        - unpack
        - open_project
        - load
        - Project

## Compatibility: `ftir_map.LoadOmnicMAP`

::: ftir_map.LoadOmnicMAP
    options:
      members:
        - Load_Omnic_Map
