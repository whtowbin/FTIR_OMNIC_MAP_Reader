# Changelog and credits

## Credits

This package would not exist without the OMNIC reader in [PyMca](https://github.com/silx-kit/pymca), the PyMca X-Ray
Fluorescence Toolkit (Copyright (c) 2004-2023 European Synchrotron Radiation Facility, V.A. Sole). PyMca is released
under the MIT license, and this package keeps its copyright notice and license text in the
[NOTICE file](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/blob/main/NOTICE). The three deprecated modules that
are modified copies of PyMca code keep their license headers.

ftir-OMNIC-map-reader is copyright (c) 2024-2026 Henry Towbin and released under the
[MIT license](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/blob/main/LICENSE.txt).

## How this reader was made

1. **2023: a repackaging of PyMca.** The first version took PyMca's `PyMca5/PyMcaIO/OmnicMap.py` and returned its
   output as xarray datasets (`Load_Omnic_Map`). All the format knowledge came from PyMca.
2. **2025: array-detector maps.** PyMca's reader was extended to read maps from array detectors, whose spectra are
   titled by pixel and stage position (`PyMca_Array_OmnicMap`).
3. **2026 (version 1.1): the directory-based reader.** Point-to-point maps failed in the PyMca-based reader, and line
   scans came back without positions. The new reader (`ftir_map.omnic`) was built using PyMca's reader as the key:
   - PyMca's constants (the offset at byte 372 minus 204, the `values[...]` and `vFloats[...]` fields of the spectral
     header, the title search, the six "map description" floats) identify the spectral header and name most of its
     fields. Following what each constant points to, in files where it works and in files where it fails, revealed
     the block directory that OMNIC files carry and the two container layouts. The comparison is in
     [File format notes, section 9](file_format.md#9-pymca-omnicmappy-constants-and-what-they-locate).
   - Spectra exported by OMNIC as CSV were found byte for byte inside the `.map` files, which pinned down where each
     spectrum's data start and how long its header is.
   - The rest (per-spectrum stage positions, video images and mosaic tiles with their stage rectangles, the map
     definition, background assignments) was worked out by comparing 81 files with OMNIC's own exports and
     screenshots, and each field is marked verified, consistent or unverified in the file format notes.

The fixes for point-to-point maps and line-scan positions were reported back to PyMca
([silx-kit/pymca#1265](https://github.com/silx-kit/pymca/issues/1265)), so that PyMca users benefit too.

## Citing and reporting problems

There is no paper for this package. If you use it in published work, cite the package by name and version
(`ftir_map.__about__.__version__`, or `m.to_xarray().attrs["reader_version"]`, which is also written into every export)
together with the repository,
[github.com/whtowbin/FTIR_OMNIC_MAP_Reader](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader).

Report a bug, a file that does not read, or a wrong value in the
[GitHub issue tracker](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/issues). For a file that does not read, say
which OMNIC version and instrument wrote it, and attach the file if you can share it.

The changelog follows, newest first.

--8<-- "CHANGELOG.md:3"
