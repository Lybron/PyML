import tensorflow as tf

# Output depth
k = 64

# Image properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(tf.float32, shape=[None, image_height, image_width, color_channels])

# Weights and bias
weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply convolution
conv_layer = tf.cnn.conv2d(input, weight, strides=[1,2,2,1], padding='SAME')

# Add bias
conv_layer = tf.cnn.add(conv_layer, bias)

# Apply activation function
conv_layer = tf.cnn.relu(conv_layer)
