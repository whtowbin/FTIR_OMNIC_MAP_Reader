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

map_path = '921005-6-plaig2-1p-array-1.map
mapdata = Load_Omnic_Map(map_path)

# %%
