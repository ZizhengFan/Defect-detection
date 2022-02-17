import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("road_defection_dick.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (3, 3))
img_log = (np.log(blur+1)/(np.log(1+np.max(blur)))) * 255
img_log = np.array(img_log, dtype=np.uint8)
bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)
edges = cv2.Canny(bilateral, 100, 200)

kernal = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernal)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)

cv2.imshow('b', img_log)
# cv2.imshow('a', featuredImg)
cv2.waitKey()
cv2.destroyAllWindows()