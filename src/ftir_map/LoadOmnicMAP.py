#%%
import numpy
import xarray as xr
# try:
#     from .PyMca_Array_OmnicMap import OmnicArrayMap
#     from .PyMca_OmnicMap import OmnicMap
# except ImportError:
#     from PyMca_Array_OmnicMap import OmnicArrayMap
#     from PyMca_OmnicMap import OmnicMap

try:
    from . import PyMca_Array_OmnicMap
    from . import PyMca_OmnicMap
except ImportError:
    import PyMca_Array_OmnicMap 
    import PyMca_OmnicMap

#%%
def Load_Omnic_Map(dir_path: str):
    try:
        mapfile = PyMca_OmnicMap.OmnicMap(dir_path)
    except Exception as e:
        print(f"Error loading map file: {e}")
        print("Trying to load as OmnicArrayMap...")
        try:
            mapfile = PyMca_Array_OmnicMap.OmnicArrayMap(dir_path)
        except Exception as e:
            print(f"Error loading map file as OmnicArrayMap: {e}")
            print("Unable to load the file.")
            return None
    
    wn0 = mapfile.info["OmnicInfo"]['First X value'] # cm^-1
    wn1 = mapfile.info["OmnicInfo"]['Last X value'] # cm^-1

    unique_x = numpy.unique(mapfile.info["positioners"]["X"])
    unique_y = numpy.unique(mapfile.info["positioners"]["Y"])

    x_coords = unique_x - unique_x[0] #µm
    y_coords = unique_y - unique_y[0] #µm

    len_wn = mapfile.data.shape[2]
    wn  = numpy.linspace(wn0, wn1, len_wn) #cm^-1

    unit_names = {"x": "um", "y": "um", "wn": "cm^-1", "data": "absorbance"}
    unit_long_names = {"x": "microns", "y": "microns", "wn": "wavenumbers", "data": "absorbance"}
    metadata = {"unit_names": unit_names, "unit_long_names": unit_long_names, **mapfile.info}


    # DataArray = xr.DataArray(
    #     mapfile.data,
    #     dims=("y", "x", "wn"),
    #     coords={"y": y_coords, "x": x_coords, "wn": wn},
    #     attrs= metadata
    # )
    #This probably needs to be a dataset to work with previous functions. or they need to be made to arrays
    lenX = len(unique_x)
    lenY = len(unique_y)
    reshaped = mapfile.data.reshape(lenY, lenX, len_wn)
    data = {"spectra": (["y", "x", "wn"], reshaped)}
    # print(f"reshaped shape: {reshaped.shape}")

    # data = {"spectra": (["y", "x", "wn"], mapfile.data)}
    # print(f"mapfile.data shape: {mapfile.data.shape}")


    coords = {
        "x": x_coords,
        "y": y_coords,
        "wn": wn,
    }

    dataset = xr.Dataset(
        data_vars= data,
        coords= coords,
        attrs= metadata
    )

    return dataset #DataArray
# %%
