# OMNIC .map file format (as read by `ftir_map.omnic`)

This page records what `ftir_map.omnic` knows about the Thermo OMNIC `.map` container. It builds on the OMNIC reader
in [PyMca](https://github.com/silx-kit/pymca) (V.A. Sole, ESRF): PyMca's byte offsets and field indices identified the
spectral header and the meaning of most of its fields, and they were the key to decoding the rest of the container.
Section 9 lists those constants and what they locate. The block directory, stage positions, images and map definition
were then worked out by comparing 81 files with OMNIC's own exports; see [How this reader was made](changelog.md#how-this-reader-was-made).

!!! warning "Not an official specification"
    Thermo has not published the `.map` format. What is not taken from PyMca was worked out by comparing files with
    OMNIC's own exports, and each item is marked verified, consistent or unverified below. A file from another OMNIC
    version or instrument may differ; `read_omnic` raises `OmnicFormatError` rather than guess.

- 78 maps from a 2017 microscope session: grids, line scans, one point-to-point map and one interferogram map.
- 3 maps from 2025: two array-detector maps and one plate map.
- Two further 2017 files are mosaic-only and hold no spectra.

Each item below is marked:

- **verified**: checked against an independent source, such as OMNIC's own CSV/SPA exports or stage positions written in the spectrum titles.
- **consistent**: holds in all 81 files, but nothing independent confirms what it means.
- **unverified**: a guess.

All numbers are little-endian.

## 1. Container

| Offset | Type | Content | Status |
|---|---|---|---|
| 0x000 | 18 bytes | magic: `Spectral Data File` (classic layout, also used by `.spa`) or `Spectral Exte File` (extended layout) | verified |
| 0x01E | char[] | file title, null-terminated | consistent |
| 0x126 | uint16 | number of directory entries | consistent |
| 0x128 | uint32 | file time, seconds since 1899-12-31 (equals the last background time on the extended maps) | consistent |
| 0x130 | entries | block directory | verified |

Directory entries:

| Layout | Size | Fields |
|---|---|---|
| classic | 16 bytes | uint16 key, uint32 offset, uint32 size, uint32 index, uint16 extra |
| extended | 22 bytes | uint16 key, uint64 offset, uint32 size, uint32 index, uint32 extra |

The `index` field means different things for different blocks:

- 1 is the map-level block.
- 0 is a background or series item.
- 1..n numbers the per-spectrum blocks in the classic layout.

### Worked example

These are the first entries of the classic point map `GCB_1C_GC52_CPX best points-50x50um_256avg.map`:

```
00000120: 0b00 0902 0800 1c00 810e 22dc 4100 0000   0x126 = 0x001c: 28 entries; 0x128 = 0xdc220e81: 2017-01-11 14:17:37
00000130: 7100 0c04 3800 0000 0000 0000 0000 0000   key 0x71, offset 0x38040c (= file size), size 0
00000140: 0200 3009 0000 8c00 0000 0100 0000 0000   key 0x02, offset 0x930, size 140, index 1: spectral header
00000150: 6a00 bc09 0000 3800 0000 0100 0000 0000   key 0x6a, offset 0x9bc, size 56, index 1
...
00000230: 9400 1460 3700 6000 0000 0100 0000 0000   key 0x94, offset 0x376014, size 96, index 1: header of spectrum 1
00000240: 0300 7460 3700 f41a 0000 0100 0000 0000   key 0x03, offset 0x376074, size 6900, index 1: spectrum 1 (1725 float32)
```

The first entry of the extended line scan `GCB_1C_GC52_CPX profile ... 256 avg.map`:

```
00000130: 0200 d036 0000 0000 0000 8c00 0000 0100 0000 0000 0000   key 0x02, offset 0x36d0 (uint64), size 140, index 1
```

## 2. Block table

| Key | Index | Size (bytes) | Content | Decoded | Status |
|---|---|---|---|---|---|
| 0x02 | 1 | 140 | spectral header of the map (section 3) | yes | verified |
| 0x02 | 0 | 140 | header of one background spectrum, or the copy at the start of 0x12d | yes | consistent |
| 0x03 | 0 | 4·n | background single-beam spectrum, float32 | yes | verified against OMNIC's background CSV export |
| 0x03 | 1..n | 4·n | classic layout: spectrum i | yes | verified (CSV and SPA exports) |
| 0x06 | 1 | 6 + 8·n + 2 | stage (x, y) of every spectrum, float32 µm | yes | verified: matches the X=/Y= titles of array maps to 0.003 µm, and the profile step sizes in the file names |
| 0x06 | 2 | 6 + 16·m + 2 | one stage rectangle (x0, y0, x1, y1) per video image or mosaic tile | yes | consistent: the count matches the image count in 81/81 files, and every image gives 0.581 × 0.589 µm/px. Orientation verified (section 8) |
| 0x06 | 3 | 6 + 4·n + 2 | per spectrum, the number of the background it was ratioed against | yes | consistent: the maximum equals the number of backgrounds − 1 (16 backgrounds on array maps, probably one per detector element) |
| 0x06 | 4 | 6 + 4·n + 2 | classic only: stage Z (focus) per spectrum, µm | yes | consistent (also stored as doubles in 0x82) |
| 0x1B | 1 | text | history ("Converted to absorbance y-axis units on ...") | yes | consistent |
| 0x1F | 1, 2 | 4·n | one float32 per spectrum (probably the value shown in OMNIC's map display) | no | unverified |
| 0x69 | 0, 1 | 12 | unknown | no | |
| 0x6A | 0, 1 | 56 | acquisition block (section 4) | partly | partly consistent |
| 0x6B | 0 | 264 | background title (256 bytes, a date string), then uint32 time (seconds since 1899) and a uint32 | yes | consistent |
| 0x71 | 0 | 0 | classic only: its offset equals the file size | no | consistent |
| 0x82 | 0 | 1188 or 1492 | map definition (section 5) | partly | consistent |
| 0x92 | 1, 2 | 728 | display settings; index 1 looks like uninitialized memory, index 2 holds the axis labels | no | unverified |
| 0x93 | 1..m | ~1.8 MB | uint32 length + BMP file: video image, 878 × 686 × 24 bit | yes | consistent |
| 0x93 | 1..m | 14460 | uint32 length + BMP thumbnail, 80 × 60, paired with each 0x14b mosaic tile | not used | consistent |
| 0x94 | 1..n | 96 | classic layout: spectrum header (section 6) | yes | verified |
| 0x12C | 1 | 1896 | descriptor of the 0x12d series (section 7) | partly | consistent |
| 0x12D | 1 | 196 + n·(100 + 4·n_wn) | extended layout: all spectra (section 6) | yes | verified (CSV and SPA exports) |
| 0x136 | 1, 2 | 1436, 168 | map annotations: an XML schema ("ThermoNicolet Map Annotation"), then label records (font "Arial", text, X/Y doubles); one file | no | |
| 0x0001 | 0 | 104 | annotation data (same record layout as 0x136/2); one file | no | |
| 0x145 | 1 | 264 | map display block: title, axis labels, and at +66 six float32 (y0, y1, dy, x0, x1, dx), PyMca's "map description" | the floats | consistent |
| 0x14B | 1..m | variable | uint32 length + JPEG file: one mosaic tile | yes | consistent |

### How the reader handles malformed blocks

- A file shorter than its header or directory, a key-2 block shorter than 140 bytes, a 0x12d block shorter than its
  196-byte prefix, an image block without its length field, or a header that says 0 points per spectrum raises
  `OmnicFormatError`.
- The key-6 count is a uint16, so a map of more than 65535 spectra cannot store all its positions. Whenever a key-6
  array (positions, background indices or Z) does not hold one entry per spectrum, the reader warns with both counts
  and ignores that array; positions become NaN. It does not truncate or pad.
- When the spectral axis is stored descending in cm⁻¹, the map's spectra and axis and every background's spectrum and
  wavenumber are reversed to ascending. Interferogram axes (data points) are kept as stored.

### Not found in any file

These were searched for as text and as float32/float64 values:

- aperture size or orientation
- detector, beamsplitter, source
- apodization, phase correction
- nominal resolution as a stored value

The only text found is the history. The aperture used appears only in the file names. Some of these parameters may be coded in the unknown fields of keys 0x02, 0x69 and 0x6A.

## 3. Spectral header (key 0x02, 140 bytes)

| Offset | Type | Field | Status |
|---|---|---|---|
| +0 | uint32 | data type: 3 spectrum, 2 background single beam, 1 interferogram | consistent |
| +4 | uint32 | number of points | verified |
| +8 | uint32 | x units: 1 cm⁻¹, 2 data points | consistent |
| +12 | uint32 | y units: 17 absorbance, 16 transmittance, 15 single beam, 22 interferogram | consistent (17 checked against CSV) |
| +16 | float32 | last X (the spectra are stored from first X to last X, ascending in cm⁻¹) | verified |
| +20 | float32 | first X | verified |
| +28 | uint32 | interferogram points collected: 8480 on the plate map named "4cmRes" and on most 2017 maps, 4384 on the 2025 array maps, 2336 on the "ultrafast" maps. The nominal resolution itself is not stored as a value | consistent |
| +32 | uint32 | interferogram peak position | per PyMca |
| +36 | uint32 | number of sample scans | verified (matches the "128 avg"/"256 avg" in file names) |
| +44 | uint32 | number of FFT points | per PyMca |
| +52 | uint32 | number of background scans | verified (the plate map says "256BkAvg") |
| +56 | float32 | background gain | per PyMca |
| +72 | float32 | "Identifier for start indices of spectra" (PyMca's name) | unverified |
| +80 | float32 | laser frequency, 15798.7 cm⁻¹ | consistent |

Notes:

- On the 2025 array map, the main header says transmittance (16) while the 0x12d series header says absorbance (17), and the history records a conversion. The reader takes `data_kind` from the series header.
- The `.spa` exports use the same header but store their data from high to low wavenumber.

## 4. Acquisition block (key 0x6A, 56 bytes)

| Offset | Content | Status |
|---|---|---|
| +4 | 1 on 2017 maps, 43 on 2025 maps | unknown |
| +20, +24 | float32 200.0 and 20000.0 in every file (possibly filter frequencies in Hz) | unverified |
| +48 | float32 2.5317, or 3.1647 on the 16 cm⁻¹ "ultrafast" maps. These are standard Nicolet mirror velocities (cm/s), so the field is exposed as `mirror_velocity` | inferred |

## 5. Map definition (key 0x82)

| Offset | Type | Content | Status |
|---|---|---|---|
| +6 | uint16 | map kind: 1 point list, 2 line, 3 area. Agrees with the kind detected from the positions in 79/79 readable files | consistent |
| +8 | 4 × float64 | x0, x1, y0, y1: area first/last column and row centers, line start/end, or the bounding box of a point list | consistent |
| +88 | float64 | step: the line step; also the area step | consistent |
| +112, +120 | float64 | x and y steps of area maps. On line and point maps these are left over from an earlier area map | consistent |
| +128, +132 | uint32 | n_x, n_y of area maps. On array maps they count detector tiles; on line and point maps they are left over | consistent |
| +164, +172 | float64 | a stage position, possibly where the background was collected | unverified |
| +180 | char[256] | map title | consistent |
| +586 | char[256] | original file name | consistent |
| +844 | char[256] | original directory on the instrument PC | consistent |
| +1146 | records | point list: uint16 0x65, three uint16, uint32 count, then count X and count Y float64 and per-point records. In the classic point map these include the six Z values | partly |

## 6. Spectra and per-spectrum headers

**Extended layout.** All spectra are in block 0x12D:

- a 196-byte prefix: a 140-byte spectral header, then a 56-byte 0x6A-like block;
- then n records, each a 96-byte header, 4 zero bytes, and 4·n_wn bytes of float32.

**Classic layout.** Each spectrum has a 0x94 header block (96 bytes) and its own 0x03 data block, with no 4-byte pad.

The 96-byte spectrum header:

| Offset | Content | Status |
|---|---|---|
| +0 | uint32 number of scans | consistent |
| +4 | uint32 time counter in centiseconds. Differences equal the time between spectra (checked against the date titles of the interferogram map); the absolute epoch is unknown | verified (differences) |
| +16 | char[64] title: "Spectrum i of n", "Pixel k, XPos=..., YPos=..., X=..., Y=..." on array maps, or a date on the interferogram map | verified |
| +88 | uint32 1 | |
| +92 | float32 (varies per spectrum) | unknown |

The stage position is not in this header; it is in key 6, index 1.

**How this was checked.** Every spectrum exported by OMNIC was found in the bytes as a run of float32 values, independently of this layout:

- 6 + 14 CSV exports and their backgrounds, little-endian, forward order;
- maximum difference 5.0e-7, which is the CSV rounding.

The matches are exactly the offsets this layout predicts:

- classic: data starts 96 bytes after each header, stride 6996 bytes;
- extended: data starts 100 bytes after each header, stride 7000 bytes.

## 7. Series descriptor (key 0x12C, 1896 bytes)

| Offset | Content |
|---|---|
| +16 | key of the series block (301 = 0x12D) |
| +36, +40, +44 | 140, 196, 292: offsets of the prefix and of the record layout inside 0x12D |
| +592 | points per spectrum |
| +596 | x-units code |
| +924 | number of spectra |
| +928 | y-units code |
| +1256 | n_x |
| +1588 | n_y (1 for line scans) |
| +1416, +1748 | axis labels ("Position (micrometers)", or "Distance (micrometers)" for line scans) |

Status: consistent.

## 8. Images and registration

Each video image (0x93) or mosaic tile (0x14B) has a stage rectangle in key 6, index 2. That gives the pixel size: 0.581 × 0.589 µm per pixel on the 2017 instrument, and 0.544 × 0.539 µm on the 2025 array map.

**Image orientation (verified).** The images need no flip or rotation. Decoded with a normal image library (which handles the bottom-up BMP row order), the first pixel row is the largest stage Y and the first pixel column is the smallest stage X. `matplotlib`'s `imshow(array, extent=(xmin, xmax, ymin, ymax))` with the default `origin="upper"` places the image correctly. `OmnicMap.to_images()` returns exactly that.

How it was checked, against OMNIC's own screenshots (which draw the video image under axes labeled in stage µm, with Y increasing upward):

- The key-6 rectangle of `GCB-S6-profile1_.JPG`'s map (x 774.87 to 1284.87, y 15133.76 to 15537.76) equals the screenshot's axis limits.
- The plot area of 11 screenshots, each with a map holding one BMP video image, was compared with the decoded image under four hypotheses: no flip, left-right, top-bottom and both. Pixel correlation with no flip was 0.88 to 0.99 and won in 11 of 11 cases. The best of the other three was 0.48 to 0.85.
- The six JPEG mosaic tiles of `GCB Scoria grain 6_map ... 128 avg.map` were placed on a canvas by their rectangles and compared with the screenshot `GCB Scoria grain 6.JPG`: correlation 0.75 with no flip, 0.45 top-bottom, about 0 for the other two. So the tiles are also upright and the n-th tile belongs to the n-th rectangle.

The test `tests/test_exports.py::test_image_orientation_against_omnic_screenshot` repeats the first comparison when `FTIR_OMNIC_MAP_DIRS` points at the Bristol 2017 folder; a synthetic bottom-up BMP covers the decoding order everywhere.

An earlier attempt that correlated image brightness with the Si-O overtone area could not decide, because the images are dark and vignetted; the screenshots settle it.

## 9. PyMca `OmnicMap.py` constants and what they locate

| Constant | What it hits in an extended map | What it hits in the classic point map |
|---|---|---|
| byte 372, uint32, minus 204 | Directory entry 3 is the 0x92 block's offset. Minus 204 gives the key-2 block + 4, because blocks 2, 0x6A and 0x69 (140 + 56 + 12 = 208 bytes) sit just before 0x92 | Lands inside the 0x145 entry and reads 0x01080000, past the end of the file. Also there is no 0x69 block, so even the right entry would be 8 bytes off. **Breaks** |
| `values[0, 6, 7, 8, 10, 12]` (13 uint32 from key2 + 4) | +4, +28, +32, +36, +44, +52 | same fields, once the block is found |
| `vFloats[0, 1, 10, 14, 16]` (from key2 + 16) | +16, +20, +56, +72, +80 | same |
| first `b"Spectrum "`, title slice of 84 bytes, `tmpValues[0, 1]` | title of record 1 in 0x12D | title in the first 0x94 block |
| channels = (distance between titles − 100) / 4 | 7000 → 1725 | 6996 → 1724. **Breaks** |
| data at title − 16 + 100 | record + 100 | 0x03 block + 4: drops the first value. **Breaks** |
| `"Position"` twice, 66 bytes apart; 6 floats at the first − 90 | 0x145 + 66 on area maps; absent on line scans, so positions come out as zeros | gap 66, but the values are point indices (1..6) |
