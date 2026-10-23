"""Load_Omnic_Map: unchanged output for grid maps, per-point output for line and point maps."""

import numpy as np
import pytest
import xarray as xr

from ftir_map.LoadOmnicMAP import Load_Omnic_Map

from .omnic_writer import write_map
from .test_omnic import BUNDLED_ARRAY_MAP, GRID, LINE, N_WN, POINTS, as_stored, spectra

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
    ds = Load_Omnic_Map(str(path))
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
    ds = Load_Omnic_Map(str(path))
    assert ds["spectra"].dims == ("point", "wn")
    np.testing.assert_array_equal(ds["spectra"].values, spectra(len(positions)))
    p = as_stored(positions)
    np.testing.assert_array_equal(ds["x_stage"].values, p[:, 0])
    np.testing.assert_allclose(ds["y"].values, p[:, 1] - p[0, 1])
    d = Load_Omnic_Map(str(path), format="Dictionary")
    assert d["data"].shape == (len(positions), N_WN)
    np.testing.assert_array_equal(d["y_stage"], p[:, 1])
    assert Load_Omnic_Map(str(path), format="Xarray-DataArray").dims == ("point", "wn")


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        Load_Omnic_Map(str(tmp_path / "missing.map"), format="csv")


def test_bundled_array_detector_map():
    d = Load_Omnic_Map(str(BUNDLED_ARRAY_MAP), format="Dictionary")
    assert d["data"].shape == (43, 31, 853)
    assert d["x"][-1] == pytest.approx(750.0, abs=0.01) and d["y"][-1] == pytest.approx(1050.0, abs=0.01)
