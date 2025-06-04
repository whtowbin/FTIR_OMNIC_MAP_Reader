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
def Load_Omnic_Map(dir_path: str, format: str = "Xarray-Dataset") -> xr.Dataset:
    """_summary_

    Args:
        dir_path (str): _description_
        format (str, optional): _description_. Defaults to "Xarray-Dataset". Other Options are "Xarray-DataArray" and "Dictionary".

    Raises:
        ValueError: Errors if the format is not supported.

    Returns:
        xr.Dataset: Dataset containing the spectral data, coordinates, and metadata.
        xr.DataArray: DataArray containing the spectral data.
        dict: Dictionary containing the spectral data, coordinates, and metadata.
    """
    try:
        mapfile = PyMca_OmnicMap.OmnicMap(dir_path)
    except Exception as e:
        print(f"Error loading map file: {e}")
        print("Trying to load as OmnicArrayMap...")
        try:
            mapfile = PyMca_Array_OmnicMap.OmnicArrayMap(dir_path)
            print("Loaded as OmnicArrayMap.")
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

    # Reshape the data to match the unique x and y coordinates
    # This assumes that the data is structured in a way that corresponds to the unique x and y coordinates.
    lenX = len(unique_x)
    lenY = len(unique_y)
    reshaped = mapfile.data.reshape(lenY, lenX, len_wn)
    data = {"spectra": (["y", "x", "wn"], reshaped)}


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
    if format == "Xarray-Dataset":
        return dataset
    elif format == "Xarray-DataArray":
        return dataset["spectra"]
    elif format == "Dictionary":
        return {
            "data": dataset["spectra"].values,
            "x": dataset["x"].values,
            "y": dataset["y"].values,
            "wn": dataset["wn"].values,
            "metadata": metadata
        }
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'Xarray-Dataset', 'Xarray-DataArray', and 'Dictionary'.")

# %%
