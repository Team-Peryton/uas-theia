# UAS Theia

Surrey Team Peryton Heron IMechE UAS 2022 Image Recognition program. Enabled successful target recognition at competition!

Unfortunatly some of the code is a mess and it's meaning will be lost to time. However, theia/main.py, theia/image_segmentation.py, theia/position_estimation.py should be understandable to a wider audience and may provide assistance to the 2023 competition provided the task does not significantly change.

We ran the code on a raspberry pi 4 using the HQ camera. This can only run the code at around 1/2 a frame per second live, so other tactics may need to be employed for successful recognition.

Target recognition from an image has been demonstrated to be very successful. However determining GPS co-ordinates from a pixel location is another kettle of fish. The position estimation code can produce inaccuracy in excess of 100m from real world data. We simplified the problem to produce a decent result for competition, however it has not been solved in the general case.

- We have provided some of the images from competition without location data : - )
- There are a bunch of problems with the code and numerous ways to improve it, however this will have to be an adventure for you!
- We modified some of the code live at comp so there's no guarantee any code will run as is.

## Sim Dataset setup

dataset download: https://www.sensefly.com/education/datasets/?dataset=1502

should be extracted into the dataset/raw_dataset directory then the datset/generate.ipynb notebook should be run
