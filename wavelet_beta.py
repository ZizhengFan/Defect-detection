import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

original = cv2.imread("lenna.png")
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
print("the shape of gray is: ", gray.shape)
# plt.imshow(gray, 'gray')
# plt.show()

coeffs2 = pywt.dwt2(gray, 'bior1.3')
# print(coeffs2[0].shape)  
LL, (LH, HL, HH) = coeffs2
print(np.max(LH))
print(np.min(LH))
print(np.max(HL))
print(np.min(HL))
print(np.max(HH))
print(np.min(HH))


threshold = 100
LH_new = pywt.threshold(data=LH, value=threshold, mode='soft', substitute=0)
HL_new = pywt.threshold(data=HL, value=threshold, mode='soft', substitute=0)
HH_new = pywt.threshold(data=HH, value=threshold, mode='soft', substitute=0)
coeff = (LL, (LH_new, HL_new, HH_new))

foo = pywt.idwt2(coeff, 'bior1.3')
plt.subplot(1, 2, 1)
plt.imshow(gray, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(foo, 'gray')
plt.show()
