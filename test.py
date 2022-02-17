import numpy as np

a = np.arange(24).reshape(4,6)
print(a)
x, y = np.where(a>10)
print((x, y))