# simple gradient descent to find the x of the function minimum

import matplotlib.pyplot as plt
import numpy as np

lr = 0.01
iter = 1000
x = -7
x_h = np.zeros (2*iter).reshape (2, iter)
for i in range (iter):
    x = x - lr * 2*(x-2) 
    x_h [0, i] = x
    x_h [1, i] = np.square (x-2) 

print (x)
plt.scatter (x_h [0], x_h [1])
plt.show()
