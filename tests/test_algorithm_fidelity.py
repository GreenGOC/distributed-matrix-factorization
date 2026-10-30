import unittest

import numpy as np

from models.dgd_local import DGDLocalAlgorithm
from models.gossip_learning import GossipNode
from network.node import DGDNode


class GossipNodePartitionTests(unittest.TestCase):
    def test_local_training_and_validation_cover_all_owned_users(self):
        ratings = np.array([[5.0, np.nan], [np.nan, 1.0]])
        node = GossipNode(
            node_id=0,
            user_ratings=ratings,
            n_items=2,
            rank=1,
            use_clipping=False,
            user_ids=np.array([10, 11]),
        )
        node.u[:] = 0.5
        node.b[:] = 0.5
        node.Y[:] = 0.5
        node.c[:] = 0.5
        self.assertAlmostEqual(node.get_validation_loss(ratings), 7.0625)

        before = node.u.copy()
        node.local_update(n_epochs=1, learning_rate=0.01, lambda_reg=0)

        self.assertTrue(np.all(np.any(node.u != before, axis=1)))
        np.testing.assert_array_equal(node.t, [1, 1])
        self.assertEqual(node.get_stats()['n_ratings'], 2)


class DGDLocalTests(unittest.TestCase):
    def test_user_partition_keeps_user_factors_local_and_consensuses_items(self):
        nodes = [
            DGDNode(0, np.array([[5.0, 0.0]]), np.array([[True, False]]), rank=1),
            DGDNode(1, np.array([[0.0, 1.0]]), np.array([[False, True]]), rank=1),
        ]
        nodes[0].Uj[:] = 0.5
        nodes[1].Uj[:] = 1.0
        nodes[0].Vj[:] = [[1.0], [2.0]]
        nodes[1].Vj[:] = [[3.0], [4.0]]
        weights = np.array([[0.75, 0.25], [0.25, 0.75]])
        expected_U = []
        expected_V = []
        snapshots = [node.Vj.copy() for node in nodes]

        for node_id, node in enumerate(nodes):
            grad_U, grad_V = node.compute_gradients(clip_norm=False)
            expected_U.append(node.Uj - 0.01 * grad_U)
            consensus_V = sum(
                weights[node_id, j] * snapshots[j] for j in range(len(nodes))
            )
            expected_V.append(consensus_V - 0.01 * grad_V)

        algorithm = DGDLocalAlgorithm(
            nodes, weights, step_size=0.01, max_iterations=1, local_epochs=1
        )
        algorithm.run_dgd_local_iteration()

        for node, user_factors, item_factors in zip(nodes, expected_U, expected_V):
            np.testing.assert_allclose(node.Uj, user_factors)
            np.testing.assert_allclose(node.Vj, item_factors)

        expected_consensus_error = np.sqrt(
            sum(np.linalg.norm(node.Vj - np.mean([n.Vj for n in nodes], axis=0)) ** 2 for node in nodes) / len(nodes)
        )
        self.assertAlmostEqual(algorithm._compute_consensus_metrics()[0], expected_consensus_error)

    def test_runner_uses_fixed_step_size_and_returns_shared_item_factors(self):
        node = DGDNode(
            0,
            np.array([[5.0, 0.0]]),
            np.array([[True, False]]),
            rank=1,
        )
        algorithm = DGDLocalAlgorithm([node], np.ones((1, 1)), 0.001, 1)

        results = algorithm.run_algorithm(
            algorithm='dgd_local',
            max_no_improvement=1,
        )

        self.assertEqual(algorithm.step_size, 0.001)
        self.assertEqual(results['V_final'].shape, node.Vj.shape)
        self.assertEqual(results['final_consensus_error'], 0.0)

    def test_unknown_algorithm_is_rejected(self):
        node = DGDNode(0, np.zeros((1, 1)), np.ones((1, 1), dtype=bool), rank=1)
        algorithm = DGDLocalAlgorithm([node], np.ones((1, 1)), 0.01, 1)

        with self.assertRaises(ValueError):
            algorithm.run_algorithm(algorithm='not-an-algorithm')


if __name__ == '__main__':
    unittest.main()
