import cv2
import os
from theia import image_segmentation
from theia.utils import display

#i'm sure a function already exists but i wanted to quickly know the accuracy of the parameters i used
files = [f for f in os.listdir('./dataset/sim_dataset/')]
unsuccessful = 0
successful = 0
for x in range(0,115):
    image = cv2.imread('./dataset/sim_dataset/' + files[x])
    square_center = image_segmentation.find_targets(image, debug=False)
    if square_center == []:
        unsuccessful += 1
    else: 
        successful += 1
print('\naccuracy was ' + str((successful/(unsuccessful+successful))*100)+ '%\n')