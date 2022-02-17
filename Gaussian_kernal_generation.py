import pywt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.filters import convolve
#---------------------------------------------------------This is a split line--
#kernel_size set (n,n) default
def gaussian_2d_kernel(kernel_size = 3,sigma = 0):
    
    kernel = np.zeros([kernel_size,kernel_size])
    center = kernel_size//2
    
    if sigma == 0:
        sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8
    
    s = 2*(sigma**2)
    sum_val = 0
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x = i-center
            y = j-center
            kernel[i,j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernel[i,j]
            #/(np.pi * s)
    sum_val = 1/sum_val
    return kernel*sum_val


def gaussian_kernel(size=5, sigma=1.0):
    """按参数生成高斯卷积核 

    Args:
        size (int, optional): [卷积核的大小]. Defaults to 5.
        sigma (float, optional): [标准差的大小 默认为1即标准正态分布]. Defaults to 1.0.

    Returns:
        [np.array]: [高斯卷积核]
    """
    # 生成二维表格 x为横向的高斯核 y为纵向的高斯核
    x, y = np.mgrid[-(size//2):(size//2)+1, -(size//2):(size//2)+1]
    # 归一化因子
    normal = 1 / (2 * np.pi * sigma**2)
    kernel = normal * np.exp(-((x**2 + y**2)/(2 * sigma**2)))

    return kernel
#---------------------------------------------------------This is a split line--
ret = gaussian_2d_kernel(5,1)
ker = gaussian_kernel(5,1)
print(ret, "\n \n", ker)