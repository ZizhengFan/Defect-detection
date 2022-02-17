import cv2
import numpy as np
from matplotlib import pyplot as plt


def adaptive_otsu(img_g):
    h, w = img_g.shape
    mask = np.zeros_like(img_g)
    winHalfWidth = 10
    localVarThresh = 0.002

    for i in range(0,w):
        new_img = img_g[:, max(1,i-winHalfWidth): min(w,i+winHalfWidth)]
        th , th_otsu = cv2.threshold(new_img, 0, 255, cv2.THRESH_OTSU)
        intile = np.var(new_img / 255)
        if intile > localVarThresh:
            _, mask[:,i:i+1] = cv2.threshold(img_g[:,i:i+1], th, 255, cv2.THRESH_BINARY)
        else:
            mask[:, i:i+1] = 255
    
    return mask

if __name__ == '__main__':
    path = "Lenna.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = adaptive_otsu(img)
    cv2.imshow('pic',img)
    cv2.waitKey()
    cv2.destroyAllWindows()




