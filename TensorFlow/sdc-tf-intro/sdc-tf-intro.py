import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World')

with tf.Session() as sess:
    # Run tf.constant operation in the session
    # 0-dimensional string tensor
    output = sess.run(hello_constant)
    print(output)

# A is a 0-dimensional int32 tensor
A = tf.constant(1234)

# B is a 1-dimensional int32 tensor
B = tf.constant([123, 456, 789])

# C is a 2-dimensional int32 tensor
C = tf.constant([[123, 456, 789], [222, 333, 444]])


# Session's feed_dict
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})

x2 = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.session() as sess:
    output = sess.run(x, feed_dict={x2:'Test String', y: 123, z: 45.67})

# Quiz
def run():
    output = None
    x3 = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x3:123)

    return output

# TensorFlow Math
# Addition - returns the sum of two parameters aas a tensor
x4 = tf.add(5, 2) # 7

# Subtraction
x5 = tf.subtract(10, 4)

# Multiplication
x6 = tf.mutiply(2, 5)

# Converting Types
# Use tf.cast() to ensure types match before performing opertaions
# e.g. tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))

# Quiz
x7 = tf.constant(10)
y7 = tf.constant(2)
z7 = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

with tf.Session() as sess:
    output = sess.run(z7)
    print(output)
