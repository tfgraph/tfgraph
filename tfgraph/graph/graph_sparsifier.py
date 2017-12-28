import numpy as np
import tensorflow as tf

from tfgraph.graph.graph import Graph


class GraphSparsifier(Graph):
  """ The graph sparsifier class implemented in the top of TensorFlow.

  This class inherits the Graph class and modifies it functionality adding a
  level of randomness on edge additions and deletions, that improves the
  performance of the results.

  Attributes:
    sess (:obj:`tf.Session`): This attribute represents the session that runs
      the TensorFlow operations.
    name (str): This attribute represents the name of the object in TensorFlow's
      op Graph.
    writer (:obj:`tf.summary.FileWriter`): This attribute represents a
      TensorFlow's Writer, that is used to obtain stats. The default value is
      `None`.
    n (int): Represents the cardinality of the vertex set as Python `int`.
    p (float): The default probability to pick an edge and add it to the
      GraphSparsifier.
    edge_count (int): Represents the cardinality of the edge set as Python `int`.
    adjacency (:obj:`tf.Tensor`): Represents the Adjacency matrix of the graph as
      2-D Tensor with shape [vertex_count,vertex_count].
    out_degrees (:obj:`tf.Tensor`): Represents the out-degrees of the
      vertices of the graph as 2-D Tensor with shape [vertex_count, 1].
    in_degrees (:obj:`tf.Tensor`): Represents the in-degrees of the vertices
      of the graph as 2-D Tensor with shape [1, vertex_count].

  """

  def __init__(self, sess: tf.Session, p: float, graph: Graph = None,
               is_sparse: bool = False, name: str = None,
               vertex_count: int = None,
               writer: tf.summary.FileWriter = None) -> None:
    """ Class Constructor of the GraphSparsfiier

    This method is called to construct a Graph object. This block of code
    initializes all the variables necessaries for this class to properly
    works.

    This class can be initialized using an edge list, that fill the graph at
    this moment, or can be construct it from the cardinality of vertices set
    given by `vertex_count` parameter.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      p (float): The default probability to pick an edge and add it to the
        GraphSparsifier.
      graph (:obj:`tfgraph.Graph`, optional): The input graph to pick the edges.
        The default value is `None`.
      writer (:obj:`tf.summary.FileWriter`, optional): This attribute represents
        a TensorFlow's Writer, that is used to obtain stats. The default value
        is `None`.
      name (str, optional): This attribute represents the name of the object in
        TensorFlow's op Graph.
      vertex_count (int, optional): Represents the cardinality of the vertex set. The
        default value is `None`.
      is_sparse (bool, optional): Use sparse Tensors if it's set to `True`. The
        default value is False` Not implemented yet. Show the Todo for more
        information.

    Todo:
      * Implement variables as sparse when it's possible. Waiting to
        TensorFlow for it.

    """
    self.p = p
    if graph is not None:

      if name is None:
        name = graph.name + "_sparsifier"

      edges_np = GraphSparsifier._sample_edges(sess, graph, p)

      Graph.__init__(self, sess, name, edges=edges_np,
                     vertex_count=graph.vertex_count,
                     is_sparse=is_sparse, writer=writer)
    else:
      Graph.__init__(self, sess, name, vertex_count=vertex_count,
                     is_sparse=is_sparse)

  @staticmethod
  def _sample_edges(sess: tf.Session, graph: Graph, p: float):
    """ Private method that sampls edges from another graph using a custom
    heuristic.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      graph (:obj:`tfgraph.Graph`): The input graph to pick the edges
      p (float): The default probability to pick an edge and add it to the
        GraphSparsifier.

    Returns:
      (:obj:`np.ndarray`): An numpy 2-D array that will contain the edges of the
        new sparsifier graph.

    """

    distribution_tf = tf.random_uniform([graph.edge_count], 0.0, 1.0)

    a = tf.reshape(tf.map_fn(
      lambda x: tf.gather(graph.out_degrees, x),
      tf.slice(graph.edge_list, [0, 0], [graph.edge_count, 1]),
      dtype=tf.float32), [graph.edge_count])

    cond_tf = p / tf.div(tf.log(np.sqrt(graph.vertex_count) + a),
                         np.log(np.sqrt(graph.vertex_count)))

    return sess.run(graph.edge_list)[sess.run(
      tf.transpose(tf.less_equal(distribution_tf, cond_tf)))]

  def append(self, src: int, dst: int):
    """ Append an edge to the graph.

    This method overrides it parent's functionality adding a certain grade of
    probability.

    This method process an input edge adding it to the graph updating all the
    variables necessaries to maintain the graph in correct state. The additions
    works with some probability, so there the addition is not guaranteed.

    Args:
      src (int): The id of the source vertex of the edge.
      dst (int): The id of the destination vertex of the edge.

    Returns:
      This method returns nothing.

    """
    distribution_tf = tf.random_uniform([1], 0.0, 1.0)

    cond_tf = self.p / tf.div(
      tf.log(np.sqrt(self.vertex_count) + self.out_degrees_vertex(src)),
      np.log(np.sqrt(self.vertex_count)))

    if self.run_tf(tf.less_equal(distribution_tf, cond_tf)):
      Graph.append(self, src, dst)

  def remove(self, src: int, dst: int):
    """ Remove an edge to the graph.

    This method overrides it parent's functionality adding a certain
    grade of probability.

    This method process an input edge deleting it to the graph updating all the
    variables necessaries to maintain the graph in correct state. The deletions
    works with some probability, so there the deletion is not guaranteed.

    Args:
      src (int): The id of the source vertex of the edge.
      dst (int): The id of the destination vertex of the edge.

    Returns:
      This method returns nothing.

    """
    distribution_tf = tf.random_uniform([1], 0.0, 1.0)

    cond_tf = self.p / tf.div(
      tf.log(tf.sqrt(self.vertex_count) + self.out_degrees_vertex(src)),
      tf.log(tf.sqrt(self.vertex_count)))

    if self.run_tf(tf.less_equal(distribution_tf, cond_tf)):
      Graph.remove(self, src, dst)
