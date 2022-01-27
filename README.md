# README #

This repository is for TalTech IAS0360 course students as a one possible final project topic. 

## What is this repository for? ##

* We describe here recordings and formats that can be used for ML model development. Also some scripts to visualize thermal readings and export JSON data into figures.

## How do I get set up? ##

* This scripts has ben tested with MATLAB 2020b.


## Scripts ##

* thermal\_visualize.m - MATLAB script to visualize thermal image from JSON file.


* thermal\_save.m - MATLAB script to save each thermal image frame into separate image file.

## Thermal JSON file format ##

Recordings are made with 10 FPS. In rare cases up to few continuous frames may appear identical wich is caused by the recording process.

* Timestamp - (double) in UNIX epoch format with millisecond precision, can be used if needed
* Sensor ID - (string) unused value
* PEC - (string) sensor error check value (unused)
* Room temperature - (float) ambient temperature measured inside the sensor
* RSSI - (integer) received signal strenth (unused)
* data - (array of floats) thermal temperature values, 32x32 array

```json
{
    "Timestamp": 1624887017.251468, 
    "Sensor ID": "Sensor_32x32_3078", 
    "PEC": "valid", 
    "Sensor size": "32x32", 
    "Room Temperature": 32.7, 
    "RSSI": -47, 
    "data": [
        [21.1, 22.4, 22.8, 21.0, 22.6, 22.8, ...], 
        [21.1, 22.4, 22.8, 21.0, 22.6, 22.8, ...], 
        ...
        [25.4, 23.9, 23.0, 21.8, 22.8, 22.9, ...], 
        [27.5, 24.2, 23.3, 23.2, 22.3, 21.5, ...]
    ]
}
```


## Thermal image file format ##

* Filename format: fig_YYYYMMDD_HHMM_SENSORID_FRAMENR_TMIN_TMAX.png
 * YYYY - year
 * MM - month
 * DD - day
 * HH - hour
 * MM - min
 * SENSORID - Sensor short unique ID (currently used "3078" and "C088")
 * FRAMENR - frame number starts from 00001 and counts up to 9999
 * TMIN - Minimum temperature (in Celsius) value detected on that frame
 * TMAX - Maximum temperature (in Celsius) value detected on that frame

## Recordings ##

### Sensor placements ###

Some of the test data was recorded in a room which has ca 15 degrees of temperature. In the middle of the room there was a bed and multple persons were walking to the sight of the sensors. Each sensor was installed to the upper corner of the wall on opposite walls.
![Sensor placements](images/sensor_placements.jpg)


### Recordings download ###

* Thermal data in raw JSON format. In total 10000 images. 5000 images/frames from each sensor.
  * [10000 thermal images](https://livettu-my.sharepoint.com/:u:/g/personal/mairo_leier_ttu_ee/ER6La16960NJnHUJAEPsmzYBV3u8DslXrxoGLQbYqre8JQ?e=eEZ1yC)
*  Thermal data converted into PNG file 875x656 pixels. This is actually 32x32 pixel data but increased for better labelling. Only for human detection.
   * [250 thermal images](https://livettu-my.sharepoint.com/:u:/g/personal/mairo_leier_ttu_ee/EQNNVwVLXfFHueyy2ccChBMBMeZWDUdZxPJU2AoBr2RUWQ?e=i1dmyS)

These links expire in 16.04.2022 (180 days after sharing).


## Hints for ML model development ##

You could consider the following information for model development:
* ML model can be developed using either Convolutional Neural Networks using thermal images or Recurrent Neural Network using raw thermal data frames.
* Each image has low and max temperature of that frame. HUman body temperature is never above 40 degrees and even with full clothing most of the time body can be detected as a bunch of warp pixels 23..25 degrees and up. 
* Body center position is changing over time while person is moving itself or moving around.
* Body size can not be larger than set of hot pixels and the closer is person to the sensor, the larger the body (number of warm close-by pixels) looks like.
