import tensorflow as tf
import numpy as np

from src.graph.graph import Graph


class GraphConstructor:
    @staticmethod
    def from_edges(sess: tf.Session, name: str, edges_np: np.ndarray,
                   writer: tf.summary.FileWriter = None,
                   is_sparse: bool = False) -> Graph:
        return Graph(sess, name, edges_np=edges_np, writer=writer,
                     is_sparse=is_sparse)

    @staticmethod
    def empty(sess: tf.Session, name: str, n: int,
              writer: tf.summary.FileWriter = None,
              sparse: bool = False) -> Graph:
        return Graph(sess, name, n=n, writer=writer, is_sparse=sparse)

    @staticmethod
    def unweighted_random(sess: tf.Session, name: str, n: int, m: int,
                          writer: tf.summary.FileWriter = None,
                          sparse: bool = False) -> Graph:
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

        return Graph(sess, name, edges_np=edges_np, writer=writer,
                     is_sparse=sparse)

    @staticmethod
    def as_naive_sparsifier(sess: tf.Session, graph: Graph, p: float,
                            sparse: bool = False) -> Graph:
        boolean_distribution = tf.less_equal(
            tf.random_uniform([graph.m], 0.0, 1.0), p)
        edges_np = graph.edge_list_np[sess.run(boolean_distribution)]
        return Graph(sess, graph.name + "_sparsifier",
                     edges_np=edges_np, is_sparse=sparse)

    @staticmethod
    def as_other_sparsifier(sess: tf.Session, graph: Graph, p: float,
                            sparse: bool = False) -> Graph:
        distribution_tf = tf.random_uniform([graph.m], 0.0, 1.0)

        v = tf.Variable(graph.out_degrees_tf)
        sess.run(tf.variables_initializer([v]))

        # print(sess.run(distribution_tf))
        a = tf.reshape(tf.map_fn(
            lambda x: tf.gather(v, x),
            tf.slice(graph.edge_list_tf, [0, 0], [graph.m, 1]),
            dtype=tf.float32), [graph.m])

        cond_tf = p / tf.div(tf.log(tf.sqrt(graph.n_tf) + a),
                             tf.log(tf.sqrt(graph.n_tf)))

        edges_np = graph.edge_list_np[sess.run(
            tf.transpose(tf.less_equal(distribution_tf, cond_tf)))]
        # print(edges_np.shape)
        return Graph(sess, graph.name + "_sparsifier", edges_np=edges_np,
                     is_sparse=sparse)

        pass
