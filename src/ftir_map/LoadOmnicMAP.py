"""`Load_Omnic_Map`: the package's original entry point, now built on `ftir_map.omnic.read_omnic`.

Grid maps (including array-detector maps whose pixels fill a grid) come back as before: spectra on (y, x, wn)
with x and y relative to the first column and row, and the PyMca-style metadata in attrs. Line scans,
point-to-point maps and irregular maps, which used to fail, come back per point on (point, wn).

This is the compatibility layer: it keeps PyMca's NaN-to-zero replacement and metadata layout so old outputs stay
identical. New code should use `ftir_map.read_omnic`, which keeps NaN as stored and offers more exports.
"""

import warnings
from typing import Literal, overload

import numpy
import xarray as xr

from .exports import grid_axis_values
from .omnic import OmnicFormatError, OmnicMap, read_omnic

FORMATS = ("Xarray-Dataset", "Xarray-DataArray", "Dictionary")
UNIT_NAMES = {"x": "um", "y": "um", "wn": "cm^-1", "data": "absorbance"}
UNIT_LONG_NAMES = {"x": "microns", "y": "microns", "wn": "wavenumbers", "data": "absorbance"}


def _omnic_info(m: OmnicMap) -> dict:
    """The keys of PyMca's OmnicMap.info["OmnicInfo"]."""
    c = m.collection
    info = {
        "Number of points": c.n_points,
        "Number of scan points": c.n_scan_points,
        "Interferogram peak position": c.interferogram_peak_position,
        "Number of sample scans": c.n_sample_scans,
        "Number of FFT points": c.n_fft_points,
        "Number of background scans": c.n_background_scans,
        "First X value": c.first_x,
        "Last X value": c.last_x,
        "Identifier for start indices of spectra": c.spectrum_set_identifier,
        "Laser frequency": c.laser_frequency,
        "Data spacing": c.data_spacing,
        "Background gain": c.background_gain,
    }
    d = m.map_description
    if d is not None:
        (x0, y0), (x1, y1) = d.first_location, d.last_location
        info["First map location"] = [x0, y0]
        info["Last map location"] = [x1, y1]
        info["Mapping stage X step size"] = d.x_step
        info["Mapping stage Y step size"] = d.y_step
        info["Number of spectra"] = abs((1 + ((y1 - y0) / d.y_step)) * (1 + ((x1 - x0) / d.x_step)))
    return info


def _legacy_info(m: OmnicMap, n_rows: int, n_columns: int) -> dict:
    """The rest of PyMca's OmnicMap.info for a grid map of n_rows x n_columns."""
    info = {"Dim_1": n_rows, "Dim_2": n_columns, "Dim_3": m.spectra.shape[1], "SourceName": [str(m.path)]}
    info.update(Size=n_rows, NumberOfFiles=n_rows, FileIndex=0, Channel0=0.0)
    info["McaCalib"] = [m.collection.first_x * 1.0, m.collection.data_spacing * 1.0, 0.0]
    info["OmnicInfo"] = _omnic_info(m)
    dx = (m.x[-1] - m.x[0]) / (n_columns - 1)
    dy = (m.y[-1] - m.y[0]) / (n_rows - 1)
    if dx > 0.0 and dy > 0.0:
        info["xScale"] = [m.x[0], dx]
        info["yScale"] = [m.y[0], dy]
    info["positioners"] = {"X": m.x.copy(), "Y": m.y.copy()}
    return info


def _grid_dataset(m: OmnicMap) -> xr.Dataset:
    if m.grid_shape is None or m.grid_index is None:
        raise ValueError(f"{m.path.name}: a {m.map_kind} map has no regular grid")
    (n_rows, n_columns), (row, col) = m.grid_shape, m.grid_index
    spectra = numpy.where(numpy.isfinite(m.spectra), m.spectra, 0.0).astype(numpy.float32)  # as PyMca did
    cube = numpy.zeros((n_rows, n_columns, spectra.shape[1]), dtype=numpy.float32)
    cube[row, col] = spectra
    xs, ys = grid_axis_values(m.x, col, n_columns), grid_axis_values(m.y, row, n_rows)
    metadata = {"unit_names": UNIT_NAMES, "unit_long_names": UNIT_LONG_NAMES, **_legacy_info(m, n_rows, n_columns)}
    coords = {"x": xs - xs[0], "y": ys - ys[0], "wn": m.axis}
    return xr.Dataset(data_vars={"spectra": (["y", "x", "wn"], cube)}, coords=coords, attrs=metadata)


def _point_dataset(m: OmnicMap) -> xr.Dataset:
    coords = {
        "point": numpy.arange(m.n_points),
        "wn": m.axis,
        "x": ("point", m.x - m.x[0]),
        "y": ("point", m.y - m.y[0]),
        "x_stage": ("point", m.x),
        "y_stage": ("point", m.y),
    }
    metadata = {"unit_names": UNIT_NAMES, "unit_long_names": UNIT_LONG_NAMES, "map_kind": m.map_kind}
    metadata.update(data_kind=m.data_kind, OmnicInfo=_omnic_info(m), SourceName=[str(m.path)])
    return xr.Dataset(data_vars={"spectra": (["point", "wn"], m.spectra)}, coords=coords, attrs=metadata)


@overload
def Load_Omnic_Map(dir_path: str, format: Literal["Xarray-Dataset"] = ...) -> xr.Dataset | None: ...
@overload
def Load_Omnic_Map(dir_path: str, format: Literal["Xarray-DataArray"]) -> xr.DataArray | None: ...
@overload
def Load_Omnic_Map(dir_path: str, format: Literal["Dictionary"]) -> dict | None: ...
@overload
def Load_Omnic_Map(dir_path: str, format: str) -> xr.Dataset | xr.DataArray | dict | None: ...


def Load_Omnic_Map(dir_path: str, format: str = "Xarray-Dataset") -> xr.Dataset | xr.DataArray | dict | None:
    """Load an OMNIC .map file.

    Args:
        dir_path (str): path to the .map file.
        format (str, optional): "Xarray-Dataset" (default), "Xarray-DataArray" or "Dictionary".

    Grid maps: spectra on (y, x, wn), x and y in um relative to the first column and row.
    Line, point and irregular maps: spectra on (point, wn), with x and y relative to the first point and the
    absolute stage positions in x_stage and y_stage; attrs["map_kind"] says which.

    Raises:
        ValueError: on an unsupported format.

    Returns:
        xr.Dataset, xr.DataArray or dict with keys data, x, y, wn and metadata (plus x_stage and y_stage for
        per-point maps). A file that cannot be read (not an OMNIC map, no spectra, missing) gives a warning and
        None, as in version 1.0. `ftir_map.read_omnic` raises OmnicFormatError instead.
    """
    if format not in FORMATS:
        raise ValueError(f"Unsupported format: {format}. Supported formats are {', '.join(map(repr, FORMATS))}.")
    try:
        m = read_omnic(dir_path)
    except (OmnicFormatError, OSError) as error:
        warnings.warn(f"Unable to load {dir_path}: {error}", stacklevel=2)
        return None
    dataset = _grid_dataset(m) if m.grid_shape is not None else _point_dataset(m)
    if format == "Xarray-Dataset":
        return dataset
    if format == "Xarray-DataArray":
        return dataset["spectra"]
    out = {"data": dataset["spectra"].values, "x": dataset["x"].values, "y": dataset["y"].values}
    if m.grid_shape is None:
        out.update(x_stage=m.x, y_stage=m.y)
    out.update(wn=dataset["wn"].values, metadata=dataset.attrs)
    return out
