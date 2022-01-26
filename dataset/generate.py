
import cv2
import numpy as np

pts1 = np.array([
    [0,0],
    [4608, 0],
    [0, 3456],
    [4608, 3456]
])
def read_positions(file_location:str, offset=75):
    # Read in the list of target locations
    with open(file_location) as f:
        lines = f.readlines()

    ref = {}
    for line in lines:
        line.replace("\n", "")
        l = line.split(",")
        ref[l[0]] = (int(l[1])+offset, int(l[2])+offset)

    return ref

ref = read_positions("./dataset/target_positions")

def warp(x_warp, y_warp, image):
    pts2 = np.array([
        [x_warp,y_warp],
        [4608-x_warp, 0],
        [0, 3456-y_warp],
        [4608, 3456]
    ])

    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    im1Reg = cv2.warpPerspective(image, h, (4608, 3456))

    return im1Reg, h


def warp_coord(centre, h):
    """ https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87 """
    return (
        int((h[0][0]*centre[0] + h[0][1]*centre[1] + h[0][2])/(h[2][0]*centre[0] + h[2][1]*centre[1] + h[2][2])),
        int((h[1][0]*centre[0] + h[1][1]*centre[1] + h[1][2])/(h[2][0]*centre[0] + h[2][1]*centre[1] + h[2][2]))
    )

    
def create_transform_images(file):
    r = []
    image = cv2.imread('./dataset/sim_dataset/'+ file)
    for i in range(0,900,300):
        for j in range(0,900,300):
            transform, h = warp(i, j, image)
            name = f"{file}_{i}-{j}.jpg"
            cv2.imwrite(f"./dataset/transform_dataset/{name}", transform)
            centre = ref[file]
            x, y = warp_coord(centre, h)
            #marked = cv2.circle(transform, (x,y), 10, (0,0,255), thickness=-1)
            #utils.display(marked)
            r.append(f"{name}, {x}, {y} \n")
    return r