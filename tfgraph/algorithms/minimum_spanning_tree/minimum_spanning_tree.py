import tensorflow as tf

from tfgraph.utils import TensorFlowObject
from tfgraph.graph import Graph


class MinimumSpanningTree(TensorFlowObject):
  """
  Attributes:
    sess (:obj:`tf.Session`): This attribute represents the session that runs
      the TensorFlow operations.
    name (str): This attribute represents the name of the object in TensorFlow's
      op Graph.
    graph (:obj:`tfgraph.Graph`): The graph on witch it will be calculated the
      algorithm. It will be treated as Directed Weighted Graph.
  """

  def __init__(self, sess: tf.Session, name: str, graph: Graph):
    """
    Constructor of the class.

    Args:
      sess (:obj:`tf.Session`): This attribute represents the session that runs
        the TensorFlow operations.
      name (str): This attribute represents the name of the object in
        TensorFlow's op Graph.
      graph (:obj:`tfgraph.Graph`): The graph on witch it will be calculated the
        algorithm. It will be treated as Directed Weighted Graph.

    """
    name = name + '_minimum_spanning_tree'
    TensorFlowObject.__init__(self, sess, name)
    self.graph = graph
    pass
