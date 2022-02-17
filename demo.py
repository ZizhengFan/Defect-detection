import cv2
import numpy as np
from matplotlib import pyplot as plt

#---------------------------------------------------------This is a split line--
def cv_show(name, image_obj) -> None:
    """using OpenCV to show a pic

    Args:
        name (str): the name of the window of this pic
        image_obj (image object): the image read from cv2.imread()
    """
    cv2.imshow(name, image_obj)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------------------------------------------This is a split line--
# 导入road defection图像，创建灰度图array
# img = cv2.imread("road_defection.png", cv2.IMREAD_COLOR)
# img = cv2.imread("road_defection.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("road_defection_linear.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("road_defection.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.imread("road_defection_dick.png", cv2.IMREAD_GRAYSCALE)

print(img.shape)

#---------------------------------------------------------This is a split line--
# 图像bgr颜色通道分离
# b, g, r = cv2.split(img)
img_log = (np.log(gray+1)/(np.log(1+np.max(gray)))) * 255
img_log = np.array(img_log, dtype=np.uint8)
#---------------------------------------------------------This is a split line--
# 二值化处理

# 普通猜数式二值法
retVal, img_binary = cv2.threshold(img_log, 225, 255, cv2.THRESH_BINARY)

# Otsu法 当图像中的目标与背景的面积相差很大时，表现为直方图没有明显的双峰，
#   或者两个峰的大小相差很大，分割效果不佳，
#   或者目标与背景的灰度有较大的重叠时也不能准确的将目标与背景分开。
# retVal, img_binary = cv2.threshold(img_log, 0, 255,
#                                    cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

# 自适应阈值分割
# img_binary = cv2.adaptiveThreshold(img, 255, 
#                                         cv2.ADAPTIVE_THRESH_MEAN_C,
#                                         cv2.THRESH_BINARY, 9, 2)

# print(retVal)

#---------------------------------------------------------This is a split line--
# 中值滤波
img_median = cv2.medianBlur(img_binary, 5)
# 高斯滤波
# img_median = cv2.GaussianBlur(img_binary, (5, 5), 1)
#---------------------------------------------------------This is a split line--
# 形态学运算 
kernal = np.ones((3, 3), np.uint8)
# 开运算
img_morpho = cv2.morphologyEx(img_median, cv2.MORPH_OPEN, kernal)
# 闭运算
# img_morpho = cv2.morphologyEx(img_median, cv2.MORPH_CLOSE, kernal)

#---------------------------------------------------------This is a split line--
# 显示区
plt.subplot(2, 2, 1)
plt.imshow(img, 'gray')
plt.title("original gray scale image")

plt.subplot(2, 2, 2)
plt.imshow(img_binary,'gray')
plt.title("image after threshold")

plt.subplot(2, 2, 3)
plt.imshow(img_median, 'gray')
plt.title("image after median filter")

plt.subplot(2, 2, 4)
plt.imshow(img_morpho, 'gray')
plt.title("image after morphology")

plt.show()

cv2.imwrite("edge_binary_road_defection.png", img_morpho)


