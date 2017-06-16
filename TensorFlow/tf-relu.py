import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

init = tf.global_variables_initializer()
# Create Model
with tf.Session() as sess:
    sess.run(init)

    hidden_layer_input = tf.add(tf.matmul(features, weights[0]), biases[0])

    relu = tf.nn.relu(hidden_layer_input)

    output = tf.add(tf.matmul(relu, weights[1]), biases[1])

    # Print session results
    print(sess.run(output))
