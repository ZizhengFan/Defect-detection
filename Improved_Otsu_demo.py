import numpy as np
import cv2
from math import floor
from matplotlib import pyplot as plt

a = np.array([[90, 130, 169],
              [4, 80, 100],
              [110, 200, 40]])

mean = floor(np.mean(a))
stdv = floor(np.std(a))
min_index = np.argmin(a)

# plt.imshow(a, 'gray')
# plt.show()

print("the mean of this subimage is: ", mean)
print("the standard deviation of this subimage is:", stdv, "\n")
# print("")
# print(a.ravel()[4])

for i in range(len(a.ravel())):
    if a.ravel()[i] < stdv:
        print("No.", i)
        print("value is", a.ravel()[i])