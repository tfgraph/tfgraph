import tensorflow as tf

from tfgraph.graph import Graph
from tfgraph.utils import TensorFlowObject


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

  def run(self) -> Graph:
    """ The run method.

    This method calculates the Minimum Spanning Tree of the graph and then
    returns it.

    Returns:
      (:obj:`tfgraph.Graph`): Graph that represents the MST.
    """
    raise NotImplementedError(
      str(self.__class__.__name__) + ' not implemented yet')
