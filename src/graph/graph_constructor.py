import tensorflow as tf
import numpy as np

from src.graph.graph import Graph


class GraphConstructor:
    @staticmethod
    def from_edges(sess, name, edges_np, writer=None):
        return Graph(sess, name, edges_np=edges_np, writer=writer)

    @staticmethod
    def empty(sess, name, n, writer=None):
        return Graph(sess, name, n=n, writer=writer)

    @staticmethod
    def unweighted_random(sess, name, n, m, writer=None):
        if m > n * (n - 1):
            raise ValueError('m would be less than n * (n - 1)')

        edges_np = np.random.random_integers(0, n - 1, [m, 2])

        cond = True
        while cond:
            # remove uniques from: https://stackoverflow.com/a/16973510/3921457
            edges_np = np.concatenate((edges_np,
                                       np.random.random_integers(0, n - 1, [
                                           m - len(edges_np), 2])), axis=0)
            _, unique_idx = np.unique(np.ascontiguousarray(edges_np).view(
                np.dtype(
                    (np.void, edges_np.dtype.itemsize * edges_np.shape[1]))),
                return_index=True)
            edges_np = edges_np[unique_idx]
            edges_np = edges_np[edges_np[:, 0] != edges_np[:, 1]]
            cond = len(edges_np) != m

        return Graph(sess, name, edges_np=edges_np, writer=writer)

    @staticmethod
    def as_naive_sparsifier(sess, graph, p):
        boolean_distribution = tf.less_equal(
            tf.random_uniform([graph.m], 0.0, 1.0), p)
        edges_np = graph.edge_list_np[sess.run(boolean_distribution)]
        return Graph(sess, graph.name + "_sparsifier",
                     edges_np=edges_np)
