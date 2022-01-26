""" 
Messy file needed in generate.ipynb for multiprocessing as well as some key functions for image manipulation
"""



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


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated_mat = cv2.warpAffine(image, M, (nW, nH))
    
    tmp = cv2.cvtColor(rotated_mat, cv2.COLOR_RGB2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(rotated_mat)
    rgba = [b,g,r,alpha]
    dst = cv2.merge(rgba,4)
    
    return dst


def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs

def noise(img):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (100, 100)) #  np.zeros((224, 224), np.float32)
    noisy_image = np.zeros(img.shape, np.float32)
    

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def increase_brightness(img, value=30):
    """ ??? %$&* """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - (value if value > 0 else 0)
    v[v > lim] = 255
    #v[v < lim] = 0
    #v[v <= lim] += value
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def imageOverlay(background, overlay, x, y, width):
    """ overlay an image and set the width of the overlay """
    
    dim = (width, width)
    # resize image
    overlay = cv2.resize(overlay, dim, interpolation = cv2.INTER_AREA)

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background