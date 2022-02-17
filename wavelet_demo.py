import cv2
import pywt
import matplotlib.pyplot as plt
import numpy as np


#---------------------------------------------------------This is a split line--
w = pywt.Wavelet('bior3.5')
mode = pywt.Modes.smooth

a = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
print(a.shape)
ca = []
cd = []

for i in range(5):
    (a, d) = pywt.dwt(a, w, mode)
    ca.append(a)
    cd.append(d)

print(a.shape)
print(d)

rec_a = []
rec_d = []

for i, coeff in enumerate(ca):
    coeff_list = [coeff, None] + [None]*i
    rec_a.append(pywt.waverec(coeff_list, w))
    
for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))


#---------------------------------------------------------This is a split line--
# # 转换灰度图像
# def rgb2gray(src):
#     assert(len(src.shape) == 3)
#     assert(src.shape[2] == 3)
#     srcf = np.float32(src)
#     dstf = np.dot(src[..., :3], [0.299, 0.587, 0.114])
#     dstf = np.clip(dstf,0,255).astype(np.uint8)
#     return dstf

# # 读取图像
# src = imageio.imread('road_defection_dick.png')
# gray = rgb2gray(src)

# w = pywt.Wavelet('bior3.5')
# # 打印bior3.5小波基信息
# print(w)

# # 对图像进行小波处理
# coeffs2 = pywt.dwt2(gray, 'bior3.5')

# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']

# ## 提取系数
# LL, (LH, HL, HH) = coeffs2

# # 显示结果
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()
