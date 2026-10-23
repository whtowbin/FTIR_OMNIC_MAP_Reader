"""Load_Omnic_Map: unchanged output for grid maps, per-point output for line and point maps."""

import subprocess
import sys
import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr

from ftir_map.LoadOmnicMAP import Load_Omnic_Map

from .omnic_writer import write_map
from .test_omnic import ARRAY_MAP, GRID, LINE, N_WN, POINTS, as_stored, real, spectra


def load_dataset(path) -> xr.Dataset:
    result = Load_Omnic_Map(str(path))
    assert isinstance(result, xr.Dataset)
    return result


def load_dict(path) -> dict:
    result = Load_Omnic_Map(str(path), format="Dictionary")
    assert isinstance(result, dict)
    return result


LEGACY_GRID_ATTRS = {
    "unit_names",
    "unit_long_names",
    "Dim_1",
    "Dim_2",
    "Dim_3",
    "SourceName",
    "Size",
    "NumberOfFiles",
    "FileIndex",
    "Channel0",
    "McaCalib",
    "OmnicInfo",
    "positioners",
}


@pytest.mark.parametrize("layout", ["classic", "extended"])
def test_grid_map_keeps_the_y_x_wn_layout(tmp_path, layout):
    path = tmp_path / "grid.map"
    write_map(path, spectra(6), GRID, layout=layout, kind="area")
    ds = load_dataset(path)
    assert isinstance(ds, xr.Dataset)
    assert ds["spectra"].dims == ("y", "x", "wn") and ds["spectra"].shape == (2, 3, N_WN)
    np.testing.assert_array_equal(ds["spectra"].values.reshape(6, N_WN), spectra(6))
    np.testing.assert_array_equal(ds["x"].values, [0.0, 25.0, 50.0])
    np.testing.assert_array_equal(ds["y"].values, [0.0, 50.0])
    assert sorted(str(c) for c in ds.coords) == ["wn", "x", "y"]
    assert LEGACY_GRID_ATTRS <= set(ds.attrs)
    assert ds.attrs["OmnicInfo"]["Number of points"] == N_WN


@pytest.mark.parametrize("layout", ["classic", "extended"])
@pytest.mark.parametrize("positions", [POINTS, LINE], ids=["points", "line"])
def test_point_and_line_maps_are_returned_per_point(tmp_path, layout, positions):
    path = tmp_path / "points.map"
    write_map(path, spectra(len(positions)), positions, layout=layout)
    ds = load_dataset(path)
    assert ds["spectra"].dims == ("point", "wn")
    np.testing.assert_array_equal(ds["spectra"].values, spectra(len(positions)))
    p = as_stored(positions)
    np.testing.assert_array_equal(ds["x_stage"].values, p[:, 0])
    np.testing.assert_allclose(ds["y"].values, p[:, 1] - p[0, 1])
    d = load_dict(path)
    assert d["data"].shape == (len(positions), N_WN)
    np.testing.assert_array_equal(d["y_stage"], p[:, 1])
    array = Load_Omnic_Map(str(path), format="Xarray-DataArray")
    assert isinstance(array, xr.DataArray) and array.dims == ("point", "wn")


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        Load_Omnic_Map(str(tmp_path / "missing.map"), format="csv")


def test_real_array_detector_map():
    d = load_dict(real(ARRAY_MAP.name))
    assert d["data"].shape == (43, 31, 853)
    assert d["x"][-1] == pytest.approx(750.0, abs=0.01) and d["y"][-1] == pytest.approx(1050.0, abs=0.01)


def test_unreadable_files_warn_and_return_none(tmp_path):
    """As in version 1.0: a file that cannot be read gives a warning and None, not an exception."""
    not_omnic = tmp_path / "x.map"
    not_omnic.write_bytes(b"6.749918e+002,1.54\n" * 40)
    mosaic_only = tmp_path / "mosaic_only.map"
    write_map(mosaic_only, spectra(4), POINTS, with_spectra=False)
    for path in (not_omnic, mosaic_only, tmp_path / "missing.map"):
        with pytest.warns(UserWarning, match="Unable to load"):
            assert Load_Omnic_Map(str(path)) is None


# ------------------------------------------------------------------ replay against the version-1.0 implementation
def open_legacy_reader(path) -> Any:
    """The PyMca-derived reader version 1.0 used: the line reader, then the array-detector reader."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ftir_map import PyMca_Array_OmnicMap, PyMca_OmnicMap
    try:
        return PyMca_OmnicMap.OmnicMap(path)
    except Exception:  # noqa: BLE001 - version 1.0 fell back to the array reader on any error
        return PyMca_Array_OmnicMap.OmnicArrayMap(path)


def legacy_load(path) -> xr.Dataset:
    """Load_Omnic_Map as it was on main (version 1.0), shaped into a Dataset."""
    mapfile = open_legacy_reader(path)
    info = mapfile.info["OmnicInfo"]
    x, y = np.unique(mapfile.info["positioners"]["X"]), np.unique(mapfile.info["positioners"]["Y"])
    n_wn = mapfile.data.shape[2]
    unit_names = {"x": "um", "y": "um", "wn": "cm^-1", "data": "absorbance"}
    unit_long_names = {"x": "microns", "y": "microns", "wn": "wavenumbers", "data": "absorbance"}
    return xr.Dataset(
        data_vars={"spectra": (["y", "x", "wn"], mapfile.data.reshape(len(y), len(x), n_wn))},
        coords={"x": x - x[0], "y": y - y[0], "wn": np.linspace(info["First X value"], info["Last X value"], n_wn)},
        attrs={"unit_names": unit_names, "unit_long_names": unit_long_names, **mapfile.info},
    )


def assert_same(a, b, where="attrs", atol=0.0):
    """Recursive equality of nested dicts, lists, arrays and scalars (NaN equals NaN) to an absolute tolerance."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{where}: keys differ"
        for key in a:
            assert_same(a[key], b[key], f"{where}[{key!r}]", atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{where}: lengths differ"
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            assert_same(u, v, f"{where}[{i}]", atol)
    elif isinstance(a, str):
        assert a == b, where
    else:
        np.testing.assert_allclose(
            np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=atol, err_msg=where
        )


# Array-detector maps: version 1.0 read the stage position from the two-decimal text in each spectrum title, this
# version from the float32 values stored in the file. They differ by up to 0.003 um (the titles are rounded).
POSITION_KEYS = ("xScale", "yScale", "positioners")


@pytest.mark.parametrize(
    ("name", "position_atol"),
    [("GCB_1C_ol54_ultrafast map_75um aperture and step.map", 0.0), (ARRAY_MAP.name, 0.01)],
    ids=["extended-grid", "array-detector"],
)
def test_grid_output_is_identical_to_version_1_0(name, position_atol):
    path = str(real(name))
    old, new = legacy_load(path), load_dataset(path)
    assert old["spectra"].dims == new["spectra"].dims and new["spectra"].dtype == old["spectra"].dtype
    np.testing.assert_array_equal(old["spectra"].values, new["spectra"].values)  # bit for bit
    np.testing.assert_array_equal(old["wn"].values, new["wn"].values)
    for coord in ("x", "y"):
        np.testing.assert_allclose(old[coord].values, new[coord].values, rtol=0, atol=max(position_atol, 1e-9))
    others = [k for k in old.attrs if k not in POSITION_KEYS]
    assert_same({k: old.attrs[k] for k in others}, {k: new.attrs[k] for k in others})
    assert_same({k: old.attrs[k] for k in POSITION_KEYS}, {k: new.attrs[k] for k in POSITION_KEYS}, atol=position_atol)


# ------------------------------------------------------------------ deprecated PyMca-derived modules
def run_python(code):
    return subprocess.run([sys.executable, "-W", "always", "-c", code], capture_output=True, text=True, check=True)


@pytest.mark.parametrize("module", ["PyMca_OmnicMap", "PyMca_Array_OmnicMap", "LoadOmnicArrayMAP"])
def test_pymca_modules_warn_and_point_to_read_omnic(module):
    result = run_python(f"import ftir_map.{module}")
    assert "DeprecationWarning" in result.stderr and "ftir_map.read_omnic" in result.stderr


def test_new_api_does_not_import_the_deprecated_modules():
    code = (
        "import sys, ftir_map, ftir_map.LoadOmnicMAP; print([m for m in sys.modules if 'PyMca' in m or 'Array' in m])"
    )
    result = run_python(code)
    assert result.stdout.strip() == "[]" and "DeprecationWarning" not in result.stderr
