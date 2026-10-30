import numpy as np

class DGDNode:
    def __init__(self, node_id, data_Yj, mask_Yj, rank, init_scale=1):
        self.node_id = node_id
        self.Yj = data_Yj
        self.mask = mask_Yj
        self.n = data_Yj.shape[0]
        self.mj = data_Yj.shape[1]
        self.rank = rank
        self.Uj = np.random.randn(self.n, rank) * init_scale
        self.Vj = np.random.randn(self.mj, rank) * init_scale
        self.Uj_prev = self.Uj.copy()
        self.Vj_prev = self.Vj.copy()
        self.neighbours = []
        self.received_U = {}
        self.received_V = {}
        self.iteration = 0
        self.gradient_history = []
        self.consensus_error_history = []
        self.lambda_adjacency = {}
        self.z_variable = np.zeros_like(self.Vj)
        self.local_model = self.Uj.copy()
        self.consensus_model = None

    def set_neighbours(self, neighbour_ids):
        self.neighbours = neighbour_ids

    def compute_gradients(self, clip_norm=True):
        prediction = self.Uj @ self.Vj.T
        residual = (prediction - self.Yj) * self.mask
        grad_U = 2.0 * residual @ self.Vj
        grad_V = 2.0 * residual.T @ self.Uj
        if clip_norm:
            grad_U_norm = np.linalg.norm(grad_U)
            if grad_U_norm > 5.0:
                grad_U = grad_U * (5.0 / grad_U_norm)
            grad_V_norm = np.linalg.norm(grad_V)
            if grad_V_norm > 5.0:
                grad_V = grad_V * (5.0 / grad_V_norm)
        gradient_norm = np.linalg.norm(grad_U) + np.linalg.norm(grad_V)
        self.gradient_history.append(gradient_norm)
        return grad_U, grad_V

    def compute_consensus_term(self, neighbour_states, weight_matrix):
        weighted_sum = np.zeros_like(self.Uj)
        for neighbour_id in neighbour_states:
            weight = weight_matrix[self.node_id, neighbour_id]
            neighbour_U = neighbour_states[neighbour_id]
            weighted_sum += weight * neighbour_U
        return weighted_sum

    def dgd_local_update(self, step_size, weight_matrix, neighbour_states):
        self.iteration += 1
        self.Uj_prev = self.Uj.copy()
        self.Vj_prev = self.Vj.copy()
        grad_U, grad_V = self.compute_gradients()
        consensus_term = self.compute_consensus_term(neighbour_states, weight_matrix)
        self.Uj = consensus_term - step_size * grad_U
        self.Vj = self.Vj - step_size * grad_V
        return self.Uj, self.Vj

    def fedavg_local_update(self, step_size, epochs):
        for _ in range(epochs):
            grad_U, grad_V = self.compute_gradients()
            self.Uj -= step_size * grad_U
            self.Vj -= step_size * grad_V
        self.local_model = self.Uj.copy()
        return self.Uj, self.Vj

    def fedprox_local_update(self, step_size, epochs, mu, global_model=None):
        if global_model is None:
            global_model = self.local_model
        for _ in range(epochs):
            grad_U, grad_V = self.compute_gradients()
            proximal_term = mu * (self.Uj - global_model)
            grad_U += proximal_term
            self.Uj -= step_size * grad_U
            self.Vj -= step_size * grad_V
        return self.Uj, self.Vj

    def send_state(self, state_type='U'):
        if state_type == 'U':
            return {'type': 'U', 'data': self.Uj.copy(), 'from': self.node_id}
        elif state_type == 'V':
            return {'type': 'V', 'data': self.Vj.copy(), 'from': self.node_id}
        elif state_type == 'consensus':
            return {'type': 'consensus', 'data': self.get_consensus_vector(), 'from': self.node_id}
        else:
            raise ValueError(f"Unknown state type: {state_type}")

    def receive_state(self, message):
        if message['type'] == 'U':
            self.received_U[message['from']] = message['data']
        elif message['type'] == 'V':
            self.received_V[message['from']] = message['data']
        elif message['type'] == 'consensus':
            self.received_U[message['from']] = message['data']

    def clear_received(self):
        self.received_U.clear()
        self.received_V.clear()

    def get_neighbour_U_states(self):
        states = {}
        for neighbour_id in self.neighbours:
            if neighbour_id in self.received_U:
                states[neighbour_id] = self.received_U[neighbour_id]
        return states

    def get_neighbour_V_states(self):
        states = {}
        for neighbour_id in self.neighbours:
            if neighbour_id in self.received_V:
                states[neighbour_id] = self.received_V[neighbour_id]
        return states

    def compute_consensus_error(self, global_U_mean):
        error = np.linalg.norm(self.get_consensus_vector() - global_U_mean)
        self.consensus_error_history.append(error)
        return error

    def get_status(self):
        status = {
            'node_id': self.node_id,
            'U_norm': np.linalg.norm(self.Uj),
            'V_norm': np.linalg.norm(self.Vj),
            'U_change': np.linalg.norm(self.Uj - self.Uj_prev),
            'V_change': np.linalg.norm(self.Vj - self.Vj_prev),
            'gradient_norm': self.gradient_history[-1] if self.gradient_history else 0.0,
            'consensus_error': self.consensus_error_history[-1] if self.consensus_error_history else 0.0,
            'iteration': self.iteration,
            'neighbours': self.neighbours.copy(),
            'n_received_U': len(self.received_U),
            'n_received_V': len(self.received_V),
            'lambda_keys': list(self.lambda_adjacency.keys())
        }
        return status

    def reset(self):
        init_scale = 0.01
        self.Uj = np.random.randn(self.n, self.rank) * init_scale
        self.Vj = np.random.randn(self.mj, self.rank) * init_scale
        self.Uj_prev = self.Uj.copy()
        self.Vj_prev = self.Vj.copy()
        self.iteration = 0
        self.clear_received()
        self.gradient_history.clear()
        self.consensus_error_history.clear()
        self.lambda_adjacency.clear()
        self.local_model = self.Uj.copy()

    def get_average_U(self):
        return np.mean(self.Uj, axis=0) if self.n > 0 else np.zeros(self.rank)

    def get_consensus_vector(self):
        return self.get_average_U()

    def set_consensus_model(self, consensus_model):
        self.consensus_model = consensus_model.copy()