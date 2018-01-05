import argparse
import sys
import tempfile

from readfile import *
from model import *

import tensorflow as tf

FLAGS = None

def main(_):
  # Check parameter
  data_dir = "/media/quang-trung/BAEED0D1EED086D3/Dataset/urban_zoning_dataset/geotagged/"

  if (len(sys.argv) == 1):
    print("Input wrong parameter: [DIR DATASET] [GENERATE_DATA / RUN_NeuralNetwork]")
    sys.exit(1)
  elif (sys.argv[1]=="GENERATE_DATA"):
    generate_k_folds(data_dir, 5)
    sys.exit(1)
  elif (sys.argv[1]!="RUN_NeuralNetwork"):
      print("Input wrong parameter: [DIR DATASET] [GENERATE_DATA / RUN_NeuralNetwork]")
      sys.exit(1)

  train_dir = data_dir + "kfolds/" + "train1.txt"
  test_dir = data_dir + "kfolds/" + "test1.txt"
  train_data = read_data(train_dir)
  #test_data = read_data(test_dir)

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
    #  x: test_data['images'], y_: test_data['labels'], keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)