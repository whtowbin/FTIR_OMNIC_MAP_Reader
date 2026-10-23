# Real OMNIC test files

These are copies of research data from Henry Towbin's GCB xenolith FTIR session (Bristol, January 2017), from the project `Ch3_Xenolith_SIMS_Clean`. They cover each container layout and map kind. Henry approves the exact set before any release.

The tests in `tests/test_omnic.py` skip any file that is missing. 
| File | Size | sha256 | Tests |
|---|---|---|---|
| `GCB_1C_GC52_CPX best points-50x50um_256avg.map` | 3.67 MB | `aebe43ebab3a60b7b8d41e01d4fcff811f5abe89471b8acd0dcc8d15f6ed3046` | classic layout ("Spectral Data File"), point-to-point map (6 points), stage Z, 2 video images |
| `GCB_1C_ol54_ultrafast map_75um aperture and step.map` | 0.83 MB | `0a09ffd5e05d982aa348f5a09407d6ea61f2cefd0e07a78c21a64e6dce95c980` | extended layout, 11 × 13 grid, 432-point spectra, 6 JPEG mosaic tiles |
| `GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map` | 0.71 MB | `c578da67dea90d206bd117ce65a6bafd22c7c644695acd2eea5d084be035e103` | extended layout, line scan (21 points, 50 µm step), 6 JPEG mosaic tiles |
| `test file GCB_1C_GC52_opx profile1_50x50um_25um step (overlapping points)_256 averages_Ifg.map` | 2.22 MB | `fe586af8c6d1155c4607d3f3ea5ddf7e7474b7e26bbe5d83bedc9dce737daae4` | interferogram map (12 × 8192 points), dates as spectrum titles |
| `921005-6-plaig2-1p-array-1.map` | 7.1 MB | `67c6f57a167be37c3f50e9acc8a493025ea1b922039dddc9035b63e648df504e` | array-detector map (2025): extended layout, 43 × 31 grid of 853-point spectra, 16 backgrounds, titles with X= and Y= positions. Moved here from `src/ftir_map/` in 1.1 so that the wheel holds no data |
| `cpx52_best_points_csv/*.CSV` (6 spectra and the background) | 7 × 50 kB | see below | OMNIC's own exports of the point map: the ground truth for the decoded spectra |

Total size: 14.5 MB. None of these files is in the wheel or the sdist (see `pyproject.toml`).

## Source paths

All paths are relative to `~/Projects/Ch3_Xenolith_SIMS_Clean/data/recovered/M11_FTIR_GCB/bristol_2017/`:

- `GCB 1_C (HT)/Center/Grain cluster 52/CPX individual points map/GCB_1C_GC52_CPX best points-50x50um_256avg.map`
- `GCB 1_C (HT)/Edge/Olivine grain 54/GCB_1C_ol54_ultrafast map_75um aperture and step.map`
- `GCB 4/Region 2/Grain 2/GCB_4_grain2_profile 2 50x100um good profile/GCB_4_Region 2_grain2_profile 2 50x100um mosaic image test.map`
- `GCB 1_C (HT)/Center/Grain cluster 52/OPX profile/raw spectra profile test/test file GCB_1C_GC52_opx profile1_50x50um_25um step (overlapping points)_256 averages_Ifg.map`
- `GCB 1_C (HT)/Center/Grain cluster 52/CPX individual points map/CPX export/individual points/*.CSV`

## CSV checksums

| File | sha256 |
|---|---|
| `GCB_1C_GC52_CPX best points-50x50um_256avg0001.CSV` | `70b5794470d2af4871e782bfca28a62884c233d8c881f452514882edc70548a2` |
| `GCB_1C_GC52_CPX best points-50x50um_256avg0002.CSV` | `38cfacadd4cfbe68ce37e407f8df11e091a94b2c59d434061eba99388b975c75` |
| `GCB_1C_GC52_CPX best points-50x50um_256avg0003.CSV` | `9a9bac55cda4c883b3197ff6a9786ebc9809e55ea7e1c33ee751f1c12dacb620` |
| `GCB_1C_GC52_CPX best points-50x50um_256avg0004.CSV` | `1162068a567c6a1b758f45713974c1ef5ee3781f63c9a9adb7d1702dc71864d9` |
| `GCB_1C_GC52_CPX best points-50x50um_256avg0005.CSV` | `8fc5e0c0d02f2572685ee99806be540cf03d77f5f2c3466361551310c5da5a7d` |
| `GCB_1C_GC52_CPX best points-50x50um_256avg0006.CSV` | `9f288b381404f7a174de278dd877f2c94ddc22e3161239714f3cd02a1be8f72a` |
| `GCB_1C_GC52_CPX best points-50x50um_256avgbkg1.CSV` | `5794647ea8595c53cf87a9b12ba6165bb38189f4b834c372843b90f2b4854417` |

## Sweeping a larger local library

To run the sweep test over more maps, set `FTIR_OMNIC_MAP_DIRS` to one or more folders, separated by `:`:

```console
FTIR_OMNIC_MAP_DIRS=/path/to/maps uv run pytest
```
