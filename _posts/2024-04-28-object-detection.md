---
title: "Hurricane Maria: Object Detection"
date: 2024-04-14
tags: [Python, YOLO, machine learning, labelme]
header:
  image: "/images/ey challenge.jpg"
excerpt: "(Python - Machine Learning) The EY Open Data Science 2024 challenge is focused on helping coastal communities become more resilient to the effects of climate change. Participants will use AI for good and help solve societal and environmental problems through technology. "
mathjax: "true"
toc: true
toc_label : "Navigate"
---
By: Michelle Kae Celine Jo-anne Bantugon<br>

Business case built by <br>
[Professor Chase Kusterer](https://github.com/chase-kusterer)<br>
Ernst & Young | NASA <br>
Hult International Business School<br>

Note: This project was a collaboration with Ella Pesola, Italo Hidalgo, Jorge Solis, Karla Marie Banal , and Marcio Pineda in the context of a Business Challenge class with Professor Chase Kusterer at Hult International Business School.

[EY Open Science Data Challenge](https://challenge.ey.com/challenges/tropical-cyclone-damage-assessment-lrrno2xm)

The 2024 challenge is focused on helping coastal communities become more resilient to the effects of climate change.<br>

How can data and AI be a lifeline for a vulnerable coastline? <br>

<img src="{{ site.url }}{{ site.baseurl }}/images/OD08.png" alt="">
[Video Presentation](https://youtu.be/5YiImxXQS6o) <br>
<img src="{{ site.url }}{{ site.baseurl }}/images/OD01.png" alt="">
<img src="{{ site.url }}{{ site.baseurl }}/images/OD02.png" alt="">


### Introduction
Located in the northeastern Caribbean, Puerto Rico is part of the "hurricane belt." The island's location puts it directly in the path of tropical storms and hurricanes that form in the Atlantic Ocean. Hurricane Maria made landfall in Puerto Rico in September 2017, with sustained winds as high as 155 mph, which was barely below the Category 5 threshold. This natural event caused considerable damage to the island's infrastructure. The entire island was affected by uprooted trees, power lines pulled down, and residential and commercial roofs being destroyed. (Scott, 2018).

In line with the above, we will analyze the Normalized Difference Vegetation Index (NDVI) to evaluate the health of vegetation pre and post storm. Moreover, the use deep learning model such as YOLO (You Only Look Once) for object detection and rapid analysis to assess infrastructure damage after hurricanes will be employed. This is crucial for efficient resource allocation and effective response in disasters' immediate aftermath. The integration of these technologies ensures that responses are more targeted and that resources are optimally used, which is crucial for areas like Puerto Rico that are frequently in the path of hurricanes.<br>

### Top Three Actionable Insights

<b>Housing Structure</b><br>
Housing structures in the Old San Juan, which is near the coastal line, are old century houses made of cobblestones or reinforced concrete with either flat roofs and or shingle roofings. The buildings were also erected near each other making them sturdier rather than stand-alone houses or buildings. While the most damaged areas by hurricane happened in rural areas where houses or buildings are more scattered, stand-alone and mostly made out of lightweight materials.<br>

One way to lessen the effect of hurricanes on buildings, be it commercial or residential is by getting people to build more hurricane-proof buildings especially in the rural areas. Based on the outreach project initiated by the Federal Emergency Management Agency (FEMA) in conjunction with the Federal Alliance for Safe homes (FLASH), strong implementation of the use of materials based on ICC standards will surely make a lot of difference. The houses, especially the older homes must ensure roof coverings are high wind-rated and attached properly regardless of the type (tiles, shingles, or metal). It is also highly recommended to install permanently-mounted hurricane shutters, strengthen the roof deck connection, and strengthen roof-to-wall connections by installing hurricane clips or straps. Lastly install a secondary water barrier and improve the anchorage of attached structures.<br>

<b>Emergency / Evacuation Plan</b><br>
The government must identify shelters and broadcast them in advance, so that people can plan their route to safety. Each house must be equipped with a Basic-Disaster kit or emergency supplies like food, water, medicine, power supplies that will last the whole family for days while waiting for rescue (OSHA). Although the aforementioned tactics are already in existence in different parts of the world, the thing here that sets them apart from other countries will be the constant education to the people of San Juan, Puerto Rico on the evacuation plan in case a hurricane hits the country again.

<b>Insurance Plan</b><br>
When Hurricane Maria hit San Juan, Puerto Rico, it uncovers that only a small percentage of the population’s homes were insured. Majority of the old homes which were passed down from generation to generation do not have insurance as well. The biggest take-away in this whole disaster was to get your homes insured as it wouldn’t be as costly as rebuilding your home or office buildings out of your own pockets (FEMA).<br>

### Importing Packages
```
# supress warnings (be careful while using this)
import warnings
warnings.filterwarnings('ignore')

# import common GIS tools
import numpy                   as np
import xarray                  as xr
import matplotlib.pyplot       as plt
import rasterio.features
import rioxarray               as rio
from matplotlib.cm import RdYlGn, Reds

# import Planetary Computer tools
import pystac_client
import planetary_computer as pc
import odc
from odc.stac import stac_load

# additional libraries
from datetime import date # date-related calculations

# GeoTiff images
import rasterio
from   osgeo import gdal

# data visualisation
from   matplotlib        import pyplot as plt
from   matplotlib.pyplot import figure
import matplotlib.image  as img
from   PIL               import Image

# model building
import ultralytics
from   ultralytics import YOLO
import labelme2yolo

# others
import os
import shutil
import zipfile
```
<b>Part I. Pre- and Post-Event NDVI Analysis </b><br>
Part I utilizes the NDVI to calculate changes in vegetation. This calculation is based on Sentinel-2 optical data and the goal is to identify areas where storms have damaged vegetation and to assess if there is significant damage to buildings in nearby regions. The process involves defining the area of interest, the time period before and after the storm, the level of detail in the imagery (30 meters for Landstat), and filtering out the clouds . Same procedures were followed in the three visualizations below, resulting in identical outputs.<br>
<b> Accessing Satellite Data </b><br>
```
## Hurricane Maria - San Juan, Puerto Rico ##

## Defining the area of interest using latitude and longitude coordinates 
## at the center of the area we would like to observe

# Define the bounding box for the selected region 
min_lon = -66.19385887
min_lat =  18.27306794
max_lon = -66.069299
max_lat =  18.400288

# setting geographic boundary
bounds = (min_lon, min_lat, max_lon, max_lat)

# setting time window surrounding the storm landfall on September 20, 2017
time_window = "2017-04-08/2017-10-31"

# calculating days in time window
print(date(2017, 10, 31) - date(2017, 4, 8 ))
```
```
## Using Planetary Computer's STAC catalog for items matching our query parameters above

# connecting to the planetary computer
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# seaching for data
search = stac.search(collections = ["sentinel-2-l2a"],
                     bbox        = bounds,
                     datetime    = time_window)

# instantiating results list
items = list(search.get_all_items())

# summarizing results
print('This is the number of scenes that touch our region:',len(items))
```
```
## Setting the resolution to 10 meters per pixel, and then convert this to degrees per pixel for the
## latitude-longitude coordinate system

# pixel resolution for the final product
resolution = 10  # meters per pixel 


# scaling to degrees per pizel
scale = resolution / 111320.0 # degrees per pixel for CRS:4326
```
```
## Use of Open Data Cube (ODC) for managing and analyzing geospatial data
## loading specific bands of data (like red, green, blue, near-infrared, and SCL)
## Filtering out clouds using the qa_pixel band and mapping at a resolution of 30 meters per pixel

xx = stac_load(
    items,
    bands      = ["red", "green", "blue", "nir", "SCL"],
    crs        = "EPSG:4326",                            # latitude-longitude
    resolution = scale,                                  # degrees
    chunks     = {"x": 2048, "y": 2048},
    dtype      = "uint16",
    patch_url  = pc.sign,
    bbox       = bounds
)
```
<b>Viewing RGB (real color) images from the time series </b><br>
```
# This will take some time to run

# subsetting results for RGB
plot_xx = xx[ ["red", "green", "blue"] ].to_array()


# showing results 
plot_xx.plot.imshow(col      = 'time', # time
                    col_wrap = 4     , # four columns per row
                    robust   = True  , 
                    vmin     = 0     ,
                    vmax     = 3000  )


# rendering results
plt.show()
```
This focuses on the lower part of San Juan where forests are prominent. It includes geographic area and time frame, relevant satellite data and various angles. Visualization aids in selecting clear satellite images, enabling assessment of the most damaged areas by comparing pre- and post-storm dates. It also helps in narrowing the time window to assess the extent of damage post-storm. A good pre- and post-storm image should have minimal cloud cover and a suitable angle for assessing NDVI changes.<br>
```
# selecting a selected time slice to view a single RGB image and the cloud mask
time_slice = 29 # October 20, 2017 (post-storm)
```
```
## plotting an RGB real color image for a single date ##

# setting plot size
fig, ax = plt.subplots( figsize = (6, 10) )


# preparing the plot
xx.isel(time = time_slice)[ ["red", "green", "blue"] ].\
    to_array().plot.imshow(robust = True,
                           ax     = ax  ,
                           vmin   = 0   ,
                           vmax   = 3000)


# titles and axis lables
ax.set_title(label = f"RGB Color Results for Scene {time_slice}")
ax.axis('off')


# rendering results
plt.show()
```
This area of interest includes tropical forests and vegetation, some of which are located near Cupey and Caimito, 
in the lower part of San Juan. These were significantly impacted as evidenced by the browning of vegetation. 
According to NASA, nearly 60% of canopy trees in the region lost branches, snapped in half, or uprooted, with 
trees that once had wide, spreading crowns now reduced to slender main trunks. Forests in Puerto Rico are now 
approximately one-third shorter on average following Hurricane Maria (NASA, 2019).<br>
<b>Applying Cloud Filtering and Masking </b><br>
```
# instantiating a colormap for SCL pixel classifications

scl_colormap = np.array(
    [
        [252,  40, 228, 255],  # 0  - NODATA - MAGENTA
        [255,   0,   4, 255],  # 1  - Saturated or Defective - RED
        [0  ,   0,   0, 255],  # 2  - Dark Areas - BLACK
        [97 ,  97,  97, 255],  # 3  - Cloud Shadow - DARK GREY
        [3  , 139,  80, 255],  # 4  - Vegetation - GREEN
        [192, 132,  12, 255],  # 5  - Bare Ground - BROWN
        [21 , 103, 141, 255],  # 6  - Water - BLUE
        [117,   0,  27, 255],  # 7  - Unclassified - MAROON
        [208, 208, 208, 255],  # 8  - Cloud - LIGHT GREY
        [244, 244, 244, 255],  # 9  - Definitely Cloud - WHITE
        [195, 231, 240, 255],  # 10 - Thin Cloud - LIGHT BLUE
        [222, 157, 204, 255],  # 11 - Snow or Ice - PINK
    ],
    dtype="uint8",
)
```
```
# function for color encoding
def colorize(xx, colormap):
    return xr.DataArray( colormap[xx.data],
                         coords = xx.coords,
                         dims   = (*xx.dims, "band") )
```
```
# filtering out water, etc.
filter_values = [0, 1, 3, 6, 8, 9, 10]

cloud_mask = ~xx.SCL.isin(filter_values) # this means not in filter_values
```
```
# appling cloud mask (filtering out clouds, cloud shadows, and water)

# storing as 16-bit integers
cleaned_data = xx.where(cloud_mask).astype("uint16")
```
```
# converting SCL to RGB
scl_rgba_clean = colorize(xx       = cleaned_data.isel(time = time_slice).SCL.compute(), 
                          colormap = scl_colormap)


# setting figure size
plt.figure(figsize = (6, 10))
plt.imshow(scl_rgba_clean)


# titles and axis labels
plt.title(label = "Cloud / Shadows / Water Mask (MAGENTA)")
plt.axis('off')


# rendering the plot
plt.show()
```
Code above creates a colormap for different land cover types based on Sentinel-2 Scene Classification Layer (SCL) 
values. It then applies the colormap to the data, highlighting vegetation, bare ground, and water in distinct colors. 
It filters out clouds, cloud shadows, and no data from the image represented by the magenta color, displaying the remaining land cover types in the area of interest.<br>
<b>Normalized Difference Vegetation Index (NDVI)</b><br>
