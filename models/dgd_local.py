import numpy as np
import time
from typing import List, Dict, Tuple

class DGDLocalAlgorithm:
    def __init__(self, nodes, weight_matrix, step_size, max_iterations, local_epochs=5):
        self.nodes = nodes
        self.J = len(nodes)
        self.weight_matrix = weight_matrix
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.local_epochs = local_epochs
        self.iteration = 0
        self.history = {
            'loss': [],
            'consensus_error': [],
            'gradient_norm': [],
            'U_diff': [],
            'time_per_iter': [],
            'communication_rounds': []
        }
        self.consensus_on_item_factors = False
        self.topology = self._build_topology_from_weights(weight_matrix)
        for j, node in enumerate(self.nodes):
            node.neighbours = self.topology[j]
            node.consensus_model = node.get_consensus_vector()
    
    def _build_topology_from_weights(self, W):
        topology = []
        for j in range(self.J):
            neighbours = []
            for i in range(self.J):
                if i != j and W[j, i] > 0:
                    neighbours.append(i)
            topology.append(neighbours)
        return topology
    
    def _exchange_messages(self):
        comm_rounds = 0
        for j in range(self.J):
            node = self.nodes[j]
            node.clear_received()
            for neighbour_id in node.neighbours:
                neighbour_node = self.nodes[neighbour_id]
                message_to_neighbour = {
                    'type': 'consensus',
                    'data': node.get_consensus_vector(),
                    'from': node.node_id
                }
                neighbour_node.receive_state(message_to_neighbour)
                message_from_neighbour = {
                    'type': 'consensus',
                    'data': neighbour_node.get_consensus_vector(),
                    'from': neighbour_node.node_id
                }
                node.receive_state(message_from_neighbour)
                comm_rounds += 2
        return comm_rounds
    
    def _get_neighbour_consensus_vectors(self, node):
        vectors = {}
        for neighbour_id in node.neighbours:
            if neighbour_id in node.received_U:
                vectors[neighbour_id] = node.received_U[neighbour_id]
        vectors[node.node_id] = node.get_consensus_vector()
        return vectors
    
    def _compute_consensus_average(self, node, neighbour_vectors):
        avg_vector = np.zeros_like(node.get_consensus_vector())
        total_weight = 0
        for neighbour_id, vector in neighbour_vectors.items():
            weight = self.weight_matrix[node.node_id, neighbour_id]
            avg_vector += weight * vector
            total_weight += weight
        if total_weight > 0:
            avg_vector = avg_vector / total_weight
        return avg_vector
    
    def _local_fedavg_training(self, node, epochs):
        for _ in range(epochs):
            grad_U, grad_V = node.compute_gradients()
            node.Uj -= self.step_size * grad_U
            node.Vj -= self.step_size * grad_V

    def run_dgd_local_iteration(self):
        self.consensus_on_item_factors = True
        start_time = time.time()
        item_factor_snapshots = [node.Vj.copy() for node in self.nodes]
        communication_rounds = int(np.count_nonzero(self.weight_matrix) - self.J)
        total_gradient_norm = 0.0

        for node in self.nodes:
            grad_U, grad_V = node.compute_gradients(clip_norm=False)
            consensus_V = np.zeros_like(node.Vj)
            for neighbour_id, neighbour_V in enumerate(item_factor_snapshots):
                consensus_V += self.weight_matrix[node.node_id, neighbour_id] * neighbour_V

            node.Uj -= self.step_size * grad_U
            node.Vj = consensus_V - self.step_size * grad_V
            total_gradient_norm += np.linalg.norm(grad_U) + np.linalg.norm(grad_V)

        avg_gradient_norm = total_gradient_norm / self.J
        loss = self._compute_global_loss()
        consensus_error, U_diff = self._compute_consensus_metrics()
        iter_time = time.time() - start_time
        self.history['loss'].append(loss)
        self.history['consensus_error'].append(consensus_error)
        self.history['gradient_norm'].append(avg_gradient_norm)
        self.history['U_diff'].append(U_diff)
        self.history['time_per_iter'].append(iter_time)
        self.history['communication_rounds'].append(communication_rounds)
        self.iteration += 1

        return {
            'iteration': self.iteration,
            'loss': loss,
            'consensus_error': consensus_error,
            'gradient_norm': avg_gradient_norm,
            'U_diff': U_diff,
            'time': iter_time,
            'communication_rounds': communication_rounds
        }
    
    def run_d_fedavg_iteration(self):
        start_time = time.time()
        comm_rounds = self._exchange_messages()
        total_gradient_norm = 0.0
        for node in self.nodes:
            self._local_fedavg_training(node, self.local_epochs)
            neighbour_vectors = self._get_neighbour_consensus_vectors(node)
            consensus_avg = self._compute_consensus_average(node, neighbour_vectors)
            for user_idx in range(node.n):
                node.Uj[user_idx] = 0.3 * node.Uj[user_idx] + 0.7 * consensus_avg
            grad_U, grad_V = node.compute_gradients()
            total_gradient_norm += np.linalg.norm(grad_U) + np.linalg.norm(grad_V)
        avg_gradient_norm = total_gradient_norm / self.J
        loss = self._compute_global_loss()
        consensus_error, U_diff = self._compute_consensus_metrics()
        iter_time = time.time() - start_time
        self.history['loss'].append(loss)
        self.history['consensus_error'].append(consensus_error)
        self.history['gradient_norm'].append(avg_gradient_norm)
        self.history['U_diff'].append(U_diff)
        self.history['time_per_iter'].append(iter_time)
        self.history['communication_rounds'].append(comm_rounds)
        self.iteration += 1

        return {
            'iteration': self.iteration,
            'loss': loss,
            'consensus_error': consensus_error,
            'gradient_norm': avg_gradient_norm,
            'U_diff': U_diff,
            'time': iter_time,
            'communication_rounds': comm_rounds
        }
    
    def run_d_fedprox_iteration(self, mu=0.01):
        start_time = time.time()
        comm_rounds = self._exchange_messages()
        total_gradient_norm = 0.0
        for node in self.nodes:
            neighbour_vectors = self._get_neighbour_consensus_vectors(node)
            consensus_avg = self._compute_consensus_average(node, neighbour_vectors)
            node.set_consensus_model(consensus_avg.reshape(1, -1).repeat(node.n, axis=0))
            for _ in range(self.local_epochs):
                grad_U, grad_V = node.compute_gradients()
                proximal_U = mu * (node.Uj - node.consensus_model)
                grad_U += proximal_U
                node.Uj -= self.step_size * grad_U
                node.Vj -= self.step_size * grad_V
            neighbour_vectors = self._get_neighbour_consensus_vectors(node)
            consensus_avg = self._compute_consensus_average(node, neighbour_vectors)
            for user_idx in range(node.n):
                node.Uj[user_idx] = 0.3 * node.Uj[user_idx] + 0.7 * consensus_avg
            grad_U, grad_V = node.compute_gradients()
            total_gradient_norm += np.linalg.norm(grad_U) + np.linalg.norm(grad_V)
        avg_gradient_norm = total_gradient_norm / self.J
        loss = self._compute_global_loss()
        consensus_error, U_diff = self._compute_consensus_metrics()
        iter_time = time.time() - start_time
        self.history['loss'].append(loss)
        self.history['consensus_error'].append(consensus_error)
        self.history['gradient_norm'].append(avg_gradient_norm)
        self.history['U_diff'].append(U_diff)
        self.history['time_per_iter'].append(iter_time)
        self.history['communication_rounds'].append(comm_rounds)
        self.iteration += 1
        return {
            'iteration': self.iteration,
            'loss': loss,
            'consensus_error': consensus_error,
            'gradient_norm': avg_gradient_norm,
            'U_diff': U_diff,
            'time': iter_time,
            'communication_rounds': comm_rounds,
            'mu': mu
        }
    
    def run_d_fedadmm_iteration(self, rho=1.0):
        start_time = time.time()
        comm_rounds = self._exchange_messages()
        total_gradient_norm = 0.0
        for node in self.nodes:
            node.Uj_prev = node.Uj.copy()
            grad_U, grad_V = node.compute_gradients()
            neighbour_vectors = self._get_neighbour_consensus_vectors(node)
            U_update = np.zeros_like(node.get_consensus_vector())
            lambda_sum = np.zeros_like(node.get_consensus_vector())
            for neighbour_id, neighbour_vec in neighbour_vectors.items():
                if neighbour_id == node.node_id:
                    continue
                lambda_key = (node.node_id, neighbour_id)
                if lambda_key not in node.lambda_adjacency:
                    node.lambda_adjacency[lambda_key] = np.zeros_like(node.get_consensus_vector())
                lambda_ij = node.lambda_adjacency[lambda_key]
                node_vec = node.get_consensus_vector()
                U_update += 0.5 * (neighbour_vec + node_vec) + (1.0/rho) * lambda_ij
                lambda_ij += (rho/2.0) * (node_vec - neighbour_vec)
                node.lambda_adjacency[lambda_key] = lambda_ij
                lambda_sum += lambda_ij
            consensus_update = (grad_U.mean(axis=0) + rho * U_update - lambda_sum) / (1.0 + rho * (len(neighbour_vectors)-1))
            for user_idx in range(node.n):
                node.Uj[user_idx] = 0.3 * node.Uj[user_idx] + 0.7 * consensus_update
            node.Vj -= self.step_size * grad_V
            total_gradient_norm += np.linalg.norm(grad_U) + np.linalg.norm(grad_V)
        avg_gradient_norm = total_gradient_norm / self.J
        loss = self._compute_global_loss()
        consensus_error, U_diff = self._compute_consensus_metrics()
        iter_time = time.time() - start_time
        self.history['loss'].append(loss)
        self.history['consensus_error'].append(consensus_error)
        self.history['gradient_norm'].append(avg_gradient_norm)
        self.history['U_diff'].append(U_diff)
        self.history['time_per_iter'].append(iter_time)
        self.history['communication_rounds'].append(comm_rounds)
        self.iteration += 1
        return {
            'iteration': self.iteration,
            'loss': loss,
            'consensus_error': consensus_error,
            'gradient_norm': avg_gradient_norm,
            'U_diff': U_diff,
            'time': iter_time,
            'communication_rounds': comm_rounds,
            'rho': rho
        }
    
    def run_algorithm(self, algorithm='d_fedavg', convergence_tol=1e-6, max_no_improvement=50, **kwargs):
        supported_algorithms = {'d_fedavg', 'd_fedprox', 'd_fedadmm', 'dgd_local'}
        if algorithm not in supported_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.consensus_on_item_factors = algorithm == 'dgd_local'
        best_loss = float('inf')
        no_improvement_count = 0
        print(f"Starting {algorithm.upper()} with {self.J} nodes")
        print(f"Step Size: {self.step_size}, Local Epochs: {self.local_epochs}")
        print("-" * 60)
        while self.iteration < self.max_iterations:
            if algorithm == 'd_fedavg':
                result = self.run_d_fedavg_iteration()
            elif algorithm == 'd_fedprox':
                mu = kwargs.get('mu', 0.01)
                result = self.run_d_fedprox_iteration(mu=mu)
            elif algorithm == 'd_fedadmm':
                rho = kwargs.get('rho', 0.01)
                result = self.run_d_fedadmm_iteration(rho=rho)
            else:
                result = self.run_dgd_local_iteration()
            if algorithm != 'dgd_local':
                if result['loss'] < best_loss - convergence_tol:
                    best_loss = result['loss']
                    no_improvement_count = 0
                    self.step_size *= 0.999
                else:
                    no_improvement_count += 1
                    self.step_size *= 1.01
            print(f"Iter {self.iteration:4d} | Loss: {result['loss']:.6e} | "
                  f"Consensus: {result['consensus_error']:.3e} | "
                  f"Comm: {result['communication_rounds']}")
            if no_improvement_count >= max_no_improvement:
                print(f"\nConverged after {self.iteration} iterations")
                break
            if result['consensus_error'] < 1e-10 and result['gradient_norm'] < 1e-8:
                print(f"\nExact consensus reached at iteration {self.iteration}")
                break

        print("-" * 60)
        print(f"Final: Iter {self.iteration}, Loss: {result['loss']:.6e}, Consensus: {result['consensus_error']:.3e}")
        return self.get_final_results()
    
    def get_final_results(self):
        all_U = np.array([node.get_consensus_vector() for node in self.nodes])
        all_V = [node.Vj for node in self.nodes]
        U_mean = np.mean(all_U, axis=0)
        V_combined = np.mean(all_V, axis=0) if self.consensus_on_item_factors else np.vstack(all_V)
        final_consensus_error, final_U_diff = self._compute_consensus_metrics()
        final_gradient_norm = self._compute_average_gradient_norm()
        results = {
            'U_final': U_mean,
            'V_final': V_combined,
            'U_local': all_U,
            'V_local': all_V,
            'final_loss': self.history['loss'][-1],
            'final_consensus_error': final_consensus_error,
            'final_gradient_norm': final_gradient_norm,
            'iterations': self.iteration,
            'history': self.history,
            'node_statuses': [node.get_status() for node in self.nodes]
        }
        return results
    
    def get_convergence_summary(self):
        if len(self.history['loss']) == 0:
            return {}
        initial_loss = self.history['loss'][0]
        final_loss = self.history['loss'][-1]
        initial_consensus = self.history['consensus_error'][0]
        final_consensus = self.history['consensus_error'][-1]
        avg_time = np.mean(self.history['time_per_iter'])
        total_comm = sum(self.history['communication_rounds'])
        summary = {
            'total_iterations': self.iteration,
            'loss_reduction': initial_loss - final_loss,
            'loss_reduction_ratio': final_loss / initial_loss,
            'consensus_improvement': initial_consensus - final_consensus,
            'final_consensus_achieved': final_consensus < 1e-10,
            'avg_iteration_time': avg_time,
            'total_communication': total_comm,
            'convergence_rate': self._estimate_convergence_rate()
        }
        return summary
    
    def _estimate_convergence_rate(self):
        if len(self.history['loss']) < 10:
            return 0.0
        losses = np.array(self.history['loss'][-50:])
        if len(losses) < 2:
            return 0.0
        x = np.arange(len(losses)).reshape(-1, 1)
        y = np.log(losses + 1e-16)
        coeffs = np.polyfit(x.flatten(), y.flatten(), 1)
        rate = -coeffs[0]
        return rate
    
    def _compute_global_loss(self):
        total_loss = 0.0
        for node in self.nodes:
            prediction = node.Uj @ node.Vj.T
            residual = (prediction - node.Yj) * node.mask
            total_loss += np.sum(residual ** 2)
        return total_loss
    
    def _compute_consensus_metrics(self):
        all_consensus = []
        for node in self.nodes:
            all_consensus.append(node.Vj if self.consensus_on_item_factors else node.get_consensus_vector())
        if len(all_consensus) == 0:
            return 0.0, 0.0
        consensus_mean = np.mean(all_consensus, axis=0)
        consensus_error = 0.0
        for vec in all_consensus:
            consensus_error += np.linalg.norm(vec - consensus_mean) ** 2
        consensus_error = np.sqrt(consensus_error / len(all_consensus))
        U_diff = 0.0
        count = 0
        for i in range(len(all_consensus)):
            for j in range(i + 1, len(all_consensus)):
                U_diff += np.linalg.norm(all_consensus[i] - all_consensus[j])
                count += 1
        U_diff = U_diff / max(count, 1)
        return consensus_error, U_diff
    
    def _compute_average_gradient_norm(self):
        total_norm = 0.0
        for node in self.nodes:
            grad_U, grad_V = node.compute_gradients(clip_norm=not self.consensus_on_item_factors)
            node_norm = np.linalg.norm(grad_U) + np.linalg.norm(grad_V)
            total_norm += node_norm
        return total_norm / self.J