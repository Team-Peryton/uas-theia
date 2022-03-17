import cv2
import os
from theia import image_segmentation
from theia.utils import display

#i'm sure a function already exists but i wanted to quickly know the accuracy of the parameters i used
files = [f for f in os.listdir('./dataset/sim_dataset/')]
unsuccessful = 0
successful = 0
options = {
    "block_size": 599,
    "c": -29,
    "ksize": 39,
    "sigma": 0,
    "epsilon": 0.02,
    "square_ar": 0.95,
    "min_area": 50,
}
for x in range(0,len(files)):
    image = cv2.imread('./dataset/sim_dataset/' + files[x])
    square_center = image_segmentation.find_targets(image, options, debug=False)
    if square_center == []:
        unsuccessful += 1
        square_center = image_segmentation.find_targets(image, options, debug=True)
    else: 
        successful += 1
accuracy=(successful/(successful+unsuccessful))
print(accuracy*100)