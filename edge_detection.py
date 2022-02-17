import cv2
import numpy as np
import math

# 图像质量检测
def psnr(imgORI, imgFIN):
    mse = np.mean( (imgORI - imgFIN) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 阈值化
def thres(imgk):
    imgbinar = cv2.adaptiveThreshold(imgk,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
    return imgbinar

# 读取图像
src = cv2.imread('111.png',0)

# 模糊化，去掉噪声
src = cv2.GaussianBlur(src,(11,11),2.0)

# 定义结构化元素
kernel = np.ones((3,3))
# 执行形态学闭运算
closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
# 执行形态学膨胀运算
dilation = cv2.dilate(closing,kernel,iterations = 1)
# 执行梯度计算，将图像膨胀的结果减去腐蚀的结果
gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
# 梯度图像取反
edges_inv = cv2.bitwise_not(gradient)
# 阈值化
imgbinary= thres(edges_inv)
# 比较图像质量
d=psnr(edges_inv,imgbinary)
if d<=psnr(src,gradient):
    # 如果原图像质量比梯度图像质量低，合并偏重边缘图像
	imgF = cv2.addWeighted(imgbinary, 0.3, edges_inv, 0.7, 0)
imgF = cv2.addWeighted(imgbinary, 0.5, edges_inv, 0.5, 0)

cv2.imshow('edge',imgF)
cv2.waitKey()
