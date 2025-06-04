#%%
# import PyMca_Array_OmnicMap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# try:
#     from .LoadOmnicMAP import Load_Omnic_Map
# except ImportError:
#     from LoadOmnicMAP import Load_Omnic_Map

try:
    # from .LoadOmnicMAP import Load_Omnic_Map
    from .LoadOmnicMAP import Load_Omnic_Map

except ImportError:
    from LoadOmnicMAP import Load_Omnic_Map
#%%

map_path = '921005-6-plaig2-1p-array-1.map'

#tests

def test_load_omnic_map_xarray():
    mapdata = Load_Omnic_Map(map_path) # Load as xarray Dataset (Default behavior)
    assert isinstance(mapdata, xr.Dataset), "Loaded data should be an xarray Dataset."
    assert "spectra" in mapdata, "Dataset should contain 'spectra' variable."
    assert isinstance(mapdata["spectra"].data, np.ndarray), "'spectra' should be a numpy array."
    assert mapdata["spectra"].ndim == 3, "'spectra' should be a 3D array."

#%%

def test_load_omnic_map_dictionary():
    mapdata = Load_Omnic_Map(map_path, format="Dictionary")
    assert isinstance(mapdata, dict), "Loaded data should be a dictionary."
    assert "data" in mapdata, "Dictionary should contain 'spectra' key."
    assert isinstance(mapdata["data"], np.ndarray), "'spectra' should be a numpy array."
    assert mapdata["data"].ndim == 3, "'spectra' should be a 3D array."

# %%

test_load_omnic_map_dictionary()
#%%
test_load_omnic_map_xarray()
# %%
