# linear regresion example 
# https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
# https://gist.github.com/sagarmainkar/41d135a04d7d3bc4098f0664fe20cf3c

import numpy as np
import matplotlib.pyplot as plt

# mean square error loss
def MSE_loss (y, l):
    n = y.size
    if n == 0:
        n = 1
    return np.sum (np.square (l-y)) / (2*n)


# mean square error derivative
def dMSE_loss (y, l):
    n = y.size
    if n == 0:
        n = 1
    return np.sum (y-l, axis=1) / n # result.shape [size, ], we sum on the axis = 1 (columns)


# gradient descent 
# J = np.sum (np.square (l-y)) / (2*n)
# y = ax + b
# chain rule: dJ / da = dJ/dy * dy/da, dJ/db = dJ/dy * dy/db
# dJ/dy = dMSE_loss = np.sum (y-l) / n, dy/da = x, dy/db = 1

def graddesc(x, l):
    lr = 0.01
    iter = 1000 
    # we have both slope and intercept in one array
    a = np.random.rand (2, 1) # a.shape [2, 1]
    # x_b keeps in the 1st row the x values for dy/da and in the 2nd row 1s for dy/da
    x_b = np.zeros (200).reshape (2, 100)
    x_b [0] = np.array(x).reshape (100)
    x_b [1] = np.full ((x_b [1].shape), 1) # x_b.shape [2, 100]
    for i in range (iter):
       c = dMSE_loss (np.dot (a.T, x_b).T, l).reshape (l.shape) # l.shape [100, 1]
       d = np.dot (x_b, c) # c.shape [100, 1]
       a = a - lr * d # a and d.shape [2,1]
    return a



# our label set
x = 2 * np.random.rand (100,1)
y = 4 + 3 * x + np.random.randn (100,1)
a = graddesc(x, y)

plt.scatter (x, y)
test_x = np.array([[0], [2]])
test_x_b = np.c_[np.ones((2,1)),test_x]
test_y = np.dot (test_x_b, a)
plt.plot(test_x, test_y, 'r-')
plt.show()
