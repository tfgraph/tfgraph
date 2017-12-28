#!/usr/bin/python3

import numpy as np
import tensorflow as tf

import tfgraph


def main():
  edges = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 1], [5, 4], [5, 6]])

  with tf.Session() as sess:
    tf_g = tfgraph.Graph(sess, name="graph", edges=edges)
    print(sess.run(tf_g.adjacency))


if __name__ == '__main__':
  main()
