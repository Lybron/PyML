"""
"""

import numpy as np
import matplotlib.pyplot as plt

# one dimensional array:
# softmax returns one-dimensional array of length 3
#scores = [3.0, 1.0, 0.2]

# two dimensional array
# softmax returns two-dimensional array of shape (3,4)
scores = np.array([[1, 2, 3, 6],
                    [2, 4, 5, 6],
                    [3, 8, 7, 6]])

def softmax(x):
    # compute softmax values for each set of scores in x
    # 1. Take exponential of the scores
    # 2. Divide by the sum of the exponential of the scores across the other categories
    return np.exp(x)/np.sum(np.exp(x), axis=0)

# probabilities sum to 1
print(softmax(scores))

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
