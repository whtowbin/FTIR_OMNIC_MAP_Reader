# FTIR OMNIC Map Reader

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Directions](#directions)
- [License](#license)

## Description

This code aims to provide a convenient interface for reading FTIR Data specifically OMNIC .MAP files. It loads OMNIC .MAP files into python as Xarray dataset objects. This provides access to the coordinates and spectra of each point in the map. 

The underlying code for reading OMINC .MAP files was part of the python package [PyMca](https://github.com/vasole/pymca?tab=readme-ov-file) X-Ray Fluorescence Toolkit. Developed by the European Synchrotron Radiation Facility.

If there is interest in adding particular features please let me know in the issues or make a pull request with the change. 

## Installation 

You will already need to have a python environment installed on your computer to install this software. 
If you do not have a python environment installed you can use [UV](https://docs.astral.sh/uv/) to install one.

To install this package you can use pip. 

```console
pip install ftir-OMNIC-map-reader
```

If you are using [UV](https://docs.astral.sh/uv/) to manage your python environment you can install the package by running the command:

```console
uv add ftir-OMNIC-map-reader
```

if you are 
If you would like to install the package from source you can do so by following these steps:
1) Download or clone this repository to your computer and save it in a location where it can remain on your computer. 
2) In the terminal navigate to the directory of the package.
3) run the command:
   - python -m pip install .
   - if you have windows you might need to run: python3 -m pip install .

## Directions
 To load a map follow this example. 

 ``` python
 from ftir_map  import LoadOmnicMAP as omnic

 map_path = "greatest_map_ever.map"
 map = omnic.Load_Omnic_Map(map_path)
 ```

 The map is an xarray [dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) object. It should contain the three main attributes
   - data_vars: This is all of the spectra in the map
   - coords: These are the spatial and spectral coordinates of the map
   - attrs: this is all of the meta data reported in the .MAP file

If you are unfamiliar with recommend looking at the xarray [tutorial](https://tutorial.xarray.dev/overview/xarray-in-45-min)

The main benefit of xarray is that it is straightforward to reference the data by spatial and spectral coordinates in the appropriate units. If you choose not to use it you should be able to access the underlying data as numpy arrays. 

I eventually hope to provide further guidance for processing the map files efficiently using xarray. 

## License

`FTIR OMNIC Map Reader` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
