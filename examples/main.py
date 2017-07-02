import tensorflow as tf
import numpy as np
import timeit

import tf_G


def main():
    beta: float = 0.85
    convergence: float = 0.01

    edges_np: np.ndarray = tf_G.DataSets.p2p_gnutella08()

    with tf.Session() as sess:
        writer: tf.summary.FileWriter = tf.summary.FileWriter(
            'logs/tensorflow/.')

        graph: tf_G.Graph = tf_G.GraphConstructor.from_edges(
            sess, "G", edges_np, writer, is_sparse=False)

        pr_alge: tf_G.PageRank = tf_G.AlgebraicPageRank(sess, "PR1",
                                                        graph,
                                                        beta)

        pr_iter: tf_G.PageRank = tf_G.IterativePageRank(
            sess, "PR1", graph, beta)
        '''
        pr_random: tf_G.PageRank = tf_G.RandomWalkPageRank(sess, "PR3",
                                                        graph,
                                                        beta)
        
        g_updateable: tf_G.Graph = tf_G.GraphConstructor.empty(sess,
                                                               "Gfollowers",
                                                               7, writer)
        pr_updateable: tf_G.PageRank = tf_G.IterativePageRank(sess,
                                                              "PRfollowers",
                                                              g_updateable,
                                                              beta)
        '''
        '''
        for r in edges_np:
            start_time = timeit.default_timer()
            g_updateable.append(r[0],r[1])
            print(timeit.default_timer() - start_time)
            print()
            writer.add_graph(sess.graph)
        g_updateable.remove(edges_np[0,0], edges_np[0,1])
        g_updateable.append(edges_np[0,0], edges_np[0,1])
        '''
        # a = pr_alge.ranks_np()
        start_time: float = timeit.default_timer()
        b: np.ndarray = pr_iter.ranks_np(convergence=convergence)
        elapsed: float = timeit.default_timer() - start_time
        print(elapsed)
        '''
        start_time = timeit.default_timer()
        c = (pr_random.ranks_np(convergence=convergence))
        elapsed = timeit.default_timer() - start_time
        print(elapsed)
        '''
        # print(a)
        # print(b)
        # print(c)
        # print((pr_alge.error_vector_compare_np(pr_iter)))
        # print((pr_alge.error_vector_compare_np(pr_random)))
        # print(pr_iter.error_vector_compare_np(pr_random))
        # print(pr_iter.ranks_np(convergence=convergence))
        # print(pr_alge.error_ranks_compare_np(pr_iter))

        g_sparse = tf_G.GraphConstructor.as_other_sparsifier(sess, graph, 0.75)

        pr_sparse = tf_G.IterativePageRank(sess, "PR_sparse", g_sparse, beta)

        start_time: float = timeit.default_timer()
        d: np.ndarray = pr_sparse.ranks_np(convergence=convergence)
        elapsed: float = timeit.default_timer() - start_time
        print(elapsed)

        print(pr_iter.error_ranks_compare_np(pr_sparse))
        print(graph.m)
        print(g_sparse.m)

        # tf_G.Utils.save_ranks("logs/csv/alge.csv",a)
        tf_G.Utils.save_ranks("logs/csv/iter.csv", b)
        # Utils.save_ranks("logs/csv/random.csv",c)
        tf_G.Utils.save_ranks("logs/csv/sparse.csv", d)
        '''
        print(GraphConstructor.unweighted_random(sess, "GRandom", 10 ** 2,
                                                 10 ** 3, writer=writer))
        '''
        writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
