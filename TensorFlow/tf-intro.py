"""
Introduction to Deep Learning with TensorFlow
Source: https://pythonprogramming.net/tensorflow-introduction-machine-learning-tutorial/
"""

import tensorflow as tf

x1 = tf.constant([[5, 3], [7, 3]])
x2 = tf.constant([[6, 1], [4, 5]])

result = tf.matmul(x1, x2)
print(result)

sess = tf.Session()
print(sess.run(result))
sess.close()

# OR

with tf.Session() as sess:
    print(sess.run(result))
