import cv2
from theia.image_segmentation import find_targets

options = {
    "block_size": 399,
    "c": -39,
    "ksize": 39,
    "sigma": 0,
    "epsilon": 0.1,
    "min_area": 1000,
    "sides": [4],
    "min_solidity": 0.6,
    "debug": True
}  # options used for square recognition

print("press a key to close the current image")
print("the target centres have been found at:")

image  = cv2.imread('target1.JPG')
print(find_targets(image, options))

image  = cv2.imread('target2.JPG')
print(find_targets(image, options))