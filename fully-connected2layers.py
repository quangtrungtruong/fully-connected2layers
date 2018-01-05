import argparse
import sys
import tempfile

import numpy
from ReadFile import ReadData

import tensorflow as tf

FLAGS = None

def deepnn(f):
  # Map the 205 features to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([205, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(f, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    #h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv, keep_prob

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def next_batch(data, batch_size, shuffle=True):
  start = data['_index_in_epoch']
  # Shuffle for the first epoch
  if data['_epochs_completed'] == 0 and start == 0 and shuffle:
    perm0 = numpy.arange(data['_num_examples'])
    numpy.random.shuffle(perm0)
    data['_images'] = data['images'][perm0]
    data['_labels'] = data['labels'][perm0]
  # Go to the next epoch
  if start + batch_size > data['_num_examples']:
    # Finished epoch
    data['_epochs_completed'] += 1
    # Get the rest examples in this epoch
    rest_num_examples = data['_num_examples'] - start
    images_rest_part = data['_images'][start:data['_num_examples']]
    labels_rest_part = data['_labels'][start:data['_num_examples']]
    # Shuffle the data
    if shuffle:
      perm = numpy.arange(data['_num_examples'])
      numpy.random.shuffle(perm)
      data['_images'] = data['images'][perm]
      data['_labels'] = data['labels'][perm]
    # Start next epoch
    start = 0
    data['_index_in_epoch'] = batch_size - rest_num_examples
    end = data['_index_in_epoch']
    images_new_part = data['_images'][start:end]
    labels_new_part = data['_labels'][start:end]
    return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
  else:
    data['_index_in_epoch'] += batch_size
    end = data['_index_in_epoch']
    return data['_images'][start:end], data['_labels'][start:end]

def main(_):
  # Check parameter
  if (len(sys.argv) == 1):
    print("Input wrong parameter: [DIR DATASET]")
    sys.exit(1)
  data_dir = "/media/quang-trung/BAEED0D1EED086D3/Dataset/urban_zoning_dataset/geotagged/" + sys.argv[1]
  train_data = ReadData(data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 205])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = next_batch(train_data, 50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #  x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)