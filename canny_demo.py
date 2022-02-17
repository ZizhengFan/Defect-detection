# import os
import cv2
import numpy as np
from scipy.ndimage.filters import convolve

# 按参数生成高斯卷积核
def gaussian_kernel(size=5, sigma=1.0):
    x, y = np.mgrid[-(size//2):(size//2)+1, -(size//2):(size//2)+1]
    normal = 1 / (2 * np.pi * sigma**2)
    kernel = normal * np.exp(-((x**2 + y**2)/(2 * sigma**2)))

    return kernel

# 高斯滤波器
def gaussian_filter(image, kernel_size=5, kernel_sigma=1.0):
    return convolve(image, gaussian_kernel(kernel_size, kernel_sigma))

# 使用Sobel算子检测水平和垂直方向梯度
def sobel_gradient(blured_img):
    sx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    sy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    sobelx =convolve(np.float32(blured_img), sx)
    sobely = convolve(np.float32(blured_img), sy)
    return sobelx,sobely

# 计算图像梯度幅值
def magnitude(sobelx,sobely):
    grad = np.sqrt(sobelx**2 + sobely**2)
    phase = cv2.phase(sobelx, sobely, 1)
    phase =(180/np.pi)*phase      #将角度转换为弧度
    x,y = np.where(grad < 10)
    phase[x,y] = 0
    grad[x,y] = 0
    return grad,phase

# 非极大值抑制
def non_max_supression(grad,phase):
    r,c = grad.shape
    new = np.zeros((r,c))
    # 储存过程值
    x1,y1 = np.where(((phase>0) & (phase<=22.5)) | ((phase>157.5) & (phase<=202.5)) | ((phase>337.5)&(phase<360)))
    x2,y2 = np.where(((phase>22.5) & (phase<=67.5)) | ((phase>202.5) & (phase<=247.5)))
    x3,y3 = np.where(((phase>67.5) & (phase<=112.5)) | ((phase>247.5) & (phase<=292.5)))
    x4,y4 = np.where(((phase>112.5) & (phase<=157.5)) | ((phase>292.5) & (phase<=337.5)))

    new[x1,y1] = 0
    new[x2,y2] = 45
    new[x3,y3] = 90
    new[x4,y4] = 135

    newgrad = np.zeros((r,c))

    #非极大值抑制
    for i in range(2,r-2):
        for j in range(2,c-2):
            if new[i,j] == 90:
                if((grad[i+1,j]<grad[i,j]) & (grad[i-1,j]<grad[i,j])):
                    newgrad[i,j]=1

            elif new[i,j] == 45:
                if((grad[i+1,j-1]<grad[i,j]) & (grad[i-1,j+1]<grad[i,j])):
                    newgrad[i,j]=1

            elif new[i,j] == 0:
                if((grad[i,j+1]<grad[i,j]) & (grad[i,j-1]<grad[i,j])):
                    newgrad[i,j]=1

            elif new[i,j] == 135:
                if((grad[i+1,j+1]<grad[i,j]) & (grad[i-1,j-1]<grad[i,j])):
                    newgrad[i,j]=1
    newgrad = np.multiply(newgrad,grad)
    return newgrad

def double_thresholding(newgrad,t_low=0.075,t_high=0.175):
    #Automating the thresholding selecting process
    r,c = newgrad.shape
    tl = t_low * np.amax(newgrad)
    th = t_high * np.amax(newgrad)

    newf = np.zeros((r,c))

    for i in range(2,r-2):
    	for j in range(2,c-2):
            if(newgrad[i,j] < tl):
                newf[i,j] = 0
            elif(newgrad[i,j]>th):
                newf[i,j] = 1
            elif( newgrad[i+1,j]>th + newgrad[i-1,j]>th + newgrad[i,j+1]>th + newgrad[i,j-1]>th + newgrad[i-1, j-1]>th + newgrad[i-1, j+1]>th + newgrad[i+1, j+1]>th + newgrad[i+1, j-1]>th):
                newf[i,j] = 1
    return np.clip(newf*255,0,255).astype(np.uint8)

def canny(image,t_low=0.075,t_high=0.175):
    assert(len(image.shape) == 2)
    blured = gaussian_filter(image)
    sobel_x,sobel_y = sobel_gradient(blured)
    grad,phase = magnitude(sobel_x,sobel_y)
    nms = non_max_supression(grad,phase)
    edge = double_thresholding(nms,t_low,t_high)
    return edge

if __name__ == "__main__":
    src = cv2.imread('road_defection.png',0)
    edge = canny(src)
    cv2.imshow('src',src)
    cv2.imshow('canny edge',edge)
    cv2.waitKey()
    cv2.destroyAllWindows()


