# UAS Theia

Surrey Team Peryton Heron IMechE UAS 2022 Image Recognition program. Enabled successful target recognition at competition!

The program aims to identify a 2x2m white square marker within a search area at the competition. The program then needs to report the GPS location of the marker to the ground station.

## How it works

The main image segmentation heavily utilises OpenCV. Each stage has been carefully chosen and tuned based on performance on our datasets. The target localisation step uses DBSCAN clustering to filter out any potentially unwanted objects/artifacts, since we know there is only one target.

![](/theia_flowchart.jpg)

The actual timing is different, for example the clustering is only run once at the end of the mission.

## Example.py

- You will need Python (3.9+) installed
- You will need to run "pip install opencv-python numpy" in your command line

```
python example.py
```

Will show the image centres of the example images as well as the threshold image and target contour.


## Other explanation

Unfortunately some of the code is a mess and it's meaning will be lost to time. However, theia/main.py, theia/image_segmentation.py, theia/position_estimation.py should be understandable to a wider audience and may provide assistance to the 2023 competition provided the task does not significantly change. A lot of the code testing and tuning was developed in experiment.ipynb, however all kinds of code atrocities have probably been committed there.

We ran the code on a raspberry pi 4 using the HQ camera. This can only run the code at around 1/2 a frame per second live, so other tactics may need to be employed for successful recognition. The tuning of the filtering etc may be dependant on your camera / lens. 

Target recognition from an image has been demonstrated to be very successful. However determining GPS co-ordinates from a pixel location is another kettle of fish. The position estimation code can produce inaccuracy in excess of 100m from real world data. We simplified the problem to produce a decent result for competition, however it has not been solved in the general case.

- We have provided some of the images from competition without location data : - )
- There are a bunch of problems with the code and numerous ways to improve it, however this will have to be an adventure for you!
- We modified some of the code live at comp so there's no guarantee any code will run as is.

![](/target1.JPG)
![](/target2.JPG)

## Full Setup

The relevant packages can be installed using pipenv, https://pipenv.pypa.io/en/latest/.

Some of the code depends on picamera2, which will only work on the raspberry pi with the HQ camera.

Some of the code depends on dronekit, which cannot be installed using pip and must be installed from their repository, or mine : ) https://github.com/o-gent/dronekit2. This can be done using "pip install ." in the library directory.

Other core dependencies include:
- numpy
- pandas
- opencv 
- sklearn
- shapely


## Sim Dataset setup

dataset download: https://www.sensefly.com/education/datasets/?dataset=1502

should be extracted into the dataset/raw_dataset directory then the datset/generate.ipynb notebook should be run
