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
    from .LoadOmnicMAP import Load_Omnic_Map

except ImportError:
    from LoadOmnicMAP import Load_Omnic_Map
#%%
# map_path = "/Users/henrytowbin/Projects/FTIR_MAP_Test/Bergell_megacryst_2025_5_6_13_41_15.map"
map_path = '/Users/henrytowbin/Projects/FTIR_MAP_Test/921005-6-plaig2-1p-array-1.map'
# mapdata = OmnicMap(map_path)
mapdata = Load_Omnic_Map(map_path)

# %%
