from typing import List
from nyx.recognition import target_recognition
from nyx.recognition.k_nearest_recognition import ocr
from nyx.utils import display, logger
import os
import cv2
import time
from cv2 import dnn_superres
from nyx.recognition import nn_ocr

dataset = r"D:\targets"

video = cv2.VideoCapture(r'D:\GH010123.MP4')

sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel("./ESPCN_x4.pb")
sr.setModel("espcn", 4)

results = []
for image in os.listdir(dataset):
    img = cv2.imread(os.path.join(dataset, image))
    #upscaled = sr.upsample(img)
    h, w = img.shape[:2]
    upscaled = cv2.resize(img,(w*4,h*4),interpolation = cv2.INTER_LINEAR)
    square = target_recognition.find_targets(upscaled)
    results.append(square)

success = True
count = 0
results = []
while success:
    try:
        _, image = video.read() 
        print(f'Read a new frame: {count}')
        upscaled = sr.upsample(image)
        r = target_recognition.find_targets(upscaled)
        logger.info(r)
        results.append(r)
    except KeyboardInterrupt:
        raise
    except:
        logger.info("frame failed")
    count += 1

# post process
results_filtered = []

for r in results:
    for r2 in r:
        results_filtered.append(r2)

for result in results_filtered:
    image = result.cropped
    
    im = cv2.resize(image, (32, 32))

    left, right = int((100-32)/2),int((100-32)/2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT,value=color)
    
    t = str(time.time()).split(".")[0] + "-" + str(time.time()).split(".")[1]
    cv2.imwrite(time.strftime(f"targets/%m-%d-%H-%M-%S-{t}.jpg"),new_im)


opts = nn_ocr.Opts(image_folder="./targets", saved_model="TPS-ResNet-BiLSTM-Attn.pth")
nn_ocr.ocr(opts)

# # ocr
# opts = nn.Opts(
#     image_folder="./targets",
#     saved_model="./TPS-ResNet-BiLSTM-Attn-case-sensitive.pth",
#     imgH=
#     )






# image = r"C:\Users\olive\Downloads\images\07 02 17_27_55 1625243275-037503.jpg"
# img = cv2.imread(image)
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
# img_thresh = cv2.adaptiveThreshold(
#     imgBlurred,
#     255,
#     cv2.ADAPTIVE_THRESH_MEAN_C, 
#     cv2.THRESH_BINARY,
#     199,
#     -25
# )
# display(img_thresh)