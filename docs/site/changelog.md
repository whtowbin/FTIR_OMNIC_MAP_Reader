# Changelog and credits

## Credits

The knowledge of the file format behind the first versions of this package comes from the OMNIC reader in
[PyMca](https://github.com/vasole/pymca) (the PyMca X-Ray Fluorescence Toolkit, Copyright (c) 2004-2023 European
Synchrotron Radiation Facility, V.A. Sole). PyMca is released under the MIT license, and the package keeps its
copyright notice and license text in the [NOTICE file](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/blob/main/NOTICE).
The three deprecated modules that are modified copies of PyMca code keep their license headers.

The rest of the format was worked out by comparing 81 `.map` files with OMNIC's own exports; see the
[File format notes](file_format.md). ftir-OMNIC-map-reader is copyright (c) 2024-2026 Henry Towbin and released under
the [MIT license](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/blob/main/LICENSE.txt).

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
