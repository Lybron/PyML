import tensorflow as tf

conv_layer = tf.cnn.conv2d(input, weight, strides=[1,2,2,1], padding='SAME')
conv_layer = tf.cnn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)

# Apply Max Pooling
conv_layer = tf.cnn.max_pool(conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
