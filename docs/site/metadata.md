# Metadata and images

## What is stored

Everything below is an attribute of the `OmnicMap` that `read_omnic` returns. The same content, as JSON-serializable
values, is in `m.to_xarray().attrs` and the last item of `m.to_numpy()`; see [Exports](exports.md).

| Attribute | Content |
|---|---|
| `m.collection` | collection parameters from the spectral header: `n_points`, `first_x`, `last_x`, `x_units`, `data_kind`, `n_scan_points`, `interferogram_peak_position`, `n_sample_scans`, `n_fft_points`, `n_background_scans`, `background_gain`, `laser_frequency`, `mirror_velocity` and `data_spacing` |
| `m.map_definition` | the map set-up: `kind` (`"area"`, `"line"` or `"points"`), `x_range`, `y_range`, `step`, `x_step`, `y_step`, `n_x`, `n_y`, `title`, `file_name` and the Windows `directory` it was saved in; `None` if the file has none |
| `m.map_description` | `first_location`, `last_location`, `x_step`, `y_step` of the map display block; `None` if absent |
| `m.backgrounds` | a list of `Background(spectrum, wavenumber, data_kind, title, time)`; `m.background_index[i]` says which one spectrum `i` was ratioed against |
| `m.titles`, `m.times_s` | title of every spectrum and its acquisition time in seconds since the first spectrum |
| `m.z` | stage Z (focus) of every spectrum, for classic-layout point maps; otherwise `None` |
| `m.history`, `m.title`, `m.file_time` | the history text, the file title and the time stored in the file |
| `m.images` | the embedded video images or mosaic tiles (see below) |

```python
from ftir_map import read_omnic

m = read_omnic("my_map.map")
print(m.collection.n_sample_scans, m.collection.n_background_scans)  # 1 1
print(f"{m.collection.first_x:.1f} to {m.collection.last_x:.1f} cm-1, {m.collection.data_spacing:.2f} cm-1 per point")
print(m.map_definition.kind, m.map_definition.n_x, m.map_definition.n_y, m.map_definition.step)  # area 11 13 75.0
print(m.titles[0], m.times_s[:3])  # Spectrum 1 of 143 [0.   0.1  0.21]
print(len(m.backgrounds), m.backgrounds[0].title, m.background_index[:3])
print(m.file_time.isoformat())  # 2017-01-12T01:17:07
```

`m.block_bytes(key)` returns the raw bytes of any directory block this reader does not decode, read from the original
file. Which fields are verified against OMNIC's own exports, and which are inferred, is marked in the
[File format notes](file_format.md). The mirror velocity, for example, is inferred.

## What OMNIC does not store

These were searched for in all 81 files used to develop the reader, as text and as numbers, and were not found:

- aperture size or orientation (it appears only in some file names),
- detector, beamsplitter and source,
- apodization and phase correction,
- the nominal resolution as a stored value.

Some of them may be coded in header fields that are not decoded yet. If you need them, record them yourself next to the
data; the reader cannot supply them.

## Images and stage extents

A map can embed the video images or mosaic tiles that OMNIC showed under the map. `m.to_images()` decodes them into a
list of `ImageArray(index, format, array, extent)`:

- `array` is a `(height, width, 3)` uint8 RGB array.
- `extent` is `(xmin, xmax, ymin, ymax)` in stage um, or `None` if the file has no rectangle for that image.
- `format` is `"BMP"` for a video image and `"JPEG"` for a mosaic tile. The tiles of one map share a coordinate
  system and are placed by their extents.

The images are upright: the first row is the largest stage Y and the first column the smallest stage X, so no flip or
rotation is needed. This was checked against OMNIC's own screenshots; see
[File format notes](file_format.md#8-images-and-registration). A map without images gives an empty list.

Showing the images under the map positions:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for image in m.to_images():
    ax.imshow(image.array, extent=image.extent)  # extent = (xmin, xmax, ymin, ymax) in um; no flip needed
ax.plot(m.x, m.y, "r.", markersize=3)
ax.set_xlabel("Stage X (um)")
ax.set_ylabel("Stage Y (um)")
plt.show()
```

This works for any map with images; a map with no images only draws the points. Stage Y increases upward in the
plot, as in OMNIC. To use the images in other programs, `m.save_images(folder)` writes them with world files that
ImageJ and QGIS read; see [Exports](exports.md#save_imagesfolder-formatpng).
