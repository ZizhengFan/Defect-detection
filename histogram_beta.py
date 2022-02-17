import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


#---------------------------------------------------------This is a split line--
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

min = np.min(img)
max = np.max(img)
hist = Counter(list(img.ravel()))
print("min: ", min, "max: ", max)

sum = 0
for key, value in hist.items():
    if key <= 35:
        sum += hist[key]

print(sum)

print(hist[34])