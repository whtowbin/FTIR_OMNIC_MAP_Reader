# FTIR Map
<!-- 
[![PyPI - Version](https://img.shields.io/pypi/v/ftir-map.svg)](https://pypi.org/project/ftir-map)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ftir-map.svg)](https://pypi.org/project/ftir-map)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install ftir-map
``` -->


This code aims to provide a convenient interface for reading FTIR Data specifically OMNIC .MAP files. It loads OMNIC .MAP files into python as Xarray dataset objects. This provides access to the coordiantes and spectra of each point in the map. 

The underlying code for reading OMINC .MAP files was part of the python package [PyMca](https://github.com/vasole/pymca?tab=readme-ov-file) X-Ray Fluorescence Toolkit. Developed by the European Synchrotron Radiation Facility


## Installation 

You will already need to have a python environemnt installed on your computer to install this software. 

To install:
1) Download or clone this respository to your computer and save it in a location where it can remain on your computer. 
2) In the terminal navigate to the directory of the package.
3) run the command:
   -  python -m pip install .
    if you have windows you might need to run python3 -m pip install .

## Directions
 To load a map follow this example. 

 ``` python
 from ftir_map  import LoadOmnicMAP as omnic

 map_path = "greatest_map_ever.map"
 map = omnic.Load_Omnic_Map(map_path)
 ```

 The map should contain the three main attributes
     data_vars: This is all of the spectra in the map
     coords: These are the spatial and spectral coordiantes of the map
     attrs: this is all of the meta data reported in the .MAP file

 You can export this data to other formats if you find those easier to work with. I eventually hope to provide further guidance for processing the map files efficiently using xarray. 

## License

`ftir-map` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
