import numpy as np
import sys
import torch
import open_clip as clip
import cv2 as cv

# print(clip.list_pretrained())

# model, _, preprocess = clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')

# model.eval()
# context_length = model.context_length
# vocab_size = model.vocab_size
#
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)

img = cv.imread(cv.samples.findFile("starry_night.jpg", cv.IMREAD_UNCHANGED))

if img is None:
    sys.exit("Could not read the image.")

if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
    image = img[:, :, ::-1]  # Reverse the last axis (BGR → RGB)

# lab_image = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# lab_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
lab_image = cv.cvtColor(img, cv.COLOR_RGB2LUV)
print(img.shape)
print()
# print("______________________LAB________________________")
print()

cv.imshow("Display window", lab_image)
# cv.imshow("Display window", img)

# order of colors is B G R
# bluepx = img[100, 100, 0]
# print(bluepx)


k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)