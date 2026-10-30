import numpy as np
from scipy import sparse

class ConsensusWeights:
    @staticmethod
    def create_doubly_stochastic(adj_matrix, method='metropolis'):
        n = adj_matrix.shape[0]
        if method == 'metropolis':
            W = np.zeros((n, n))
            degrees = np.sum(adj_matrix, axis=1)
            for i in range(n):
                neighbours = np.where(adj_matrix[i] == 1)[0]
                for j in neighbours:
                    if i != j:
                        W[i, j] = 1.0 / (max(degrees[i], degrees[j]) + 1)
                W[i, i] = 1.0 - np.sum(W[i, :])
        elif method == 'max-degree':
            d_max = np.max(np.sum(adj_matrix, axis=1))
            W = adj_matrix.copy().astype(float) / (d_max + 1)
            np.fill_diagonal(W, 1.0 - np.sum(W, axis=1))
        else:
            raise ValueError(f"Unknown Method {method}")
        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)
        if not np.allclose(row_sums, 1.0) or not np.allclose(col_sums, 1.0):
            pass
        return W
    
    @staticmethod
    def ensure_omega_condition(W, strategy='shift_to_self'):
        n = W.shape[0]
        if strategy == 'shift_to_self':
            I = np.eye(n)
            W_tilde = (W + I) / 2.0
        elif strategy == 'scale':
            alpha = 0.6
            W_tilde = alpha * W + (1.0 - alpha) * np.eye(n)
        else:
            raise ValueError(f"Unknown Strategy {strategy}")
        off_diag_sums = np.sum(W_tilde, axis=1) - np.diag(W_tilde)
        omega = np.max(off_diag_sums)
        return W_tilde, omega
    
    @staticmethod
    def compute_consensus_stepsize(W_tilde, lipschitz_estimate, safety_factor=0.8):
        off_diag_sums = np.sum(W_tilde, axis=1) - np.diag(W_tilde)
        omega = np.max(off_diag_sums)
        mu_max = (1.0 - 2.0 * omega) / lipschitz_estimate
        mu_safe = safety_factor * mu_max
        return mu_max, mu_safe
    
    @staticmethod
    def analyse_weight_matrix(W, name="Weight Matrix"):
        print(f"\n{name} Analysis:")
        print("-" * 40)
        print(f"Shape: {W.shape}")
        print(f"Symmetric: {np.allclose(W, W.T)}")
        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)
        print(f"Row sums: [{np.min(row_sums):.3f}, {np.max(row_sums):.3f}]")
        print(f"Column sums: [{np.min(col_sums):.3f}, {np.max(col_sums):.3f}]")
        eigenvalues = np.linalg.eigvals(W)
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
        print(f"Largest eigenvalue magnitude: {sorted_eigs[0]:.4f}")
        print(f"Second largest eigenvalue magnitude: {sorted_eigs[1]:.4f}")
        off_diag_sums = np.sum(W, axis=1) - np.diag(W)
        omega = np.max(off_diag_sums)
        print(f"ω = max_j ∑_i!=j w_ji = {omega:.4f}")
        print(f"Condition ω < 0.5: {'✓' if omega < 0.5 else '✗'}")
        return {
            'omega': omega,
            'eigenvalues': eigenvalues,
            'row_sums': row_sums,
            'col_sums': col_sums
        }
    
    @staticmethod
    def create_fl_weight_matrix(adj_matrix, algorithm='fedavg'):
        n = adj_matrix.shape[0]
        if algorithm == 'fedavg':
            W = adj_matrix.copy().astype(float)
            degrees = np.sum(adj_matrix, axis=1)
            for i in range(n):
                if degrees[i] > 0:
                    W[i, :] = W[i, :] / degrees[i]
                W[i, i] = 1.0
        elif algorithm == 'fedprox':
            W = adj_matrix.copy().astype(float)
            for i in range(n):
                neighbours = np.where(adj_matrix[i] == 1)[0]
                total_weight = len(neighbours) + 1.0
                W[i, i] = 0.7
                for j in neighbours:
                    W[i, j] = 0.3 / len(neighbours)
        elif algorithm == 'fedadmm':
            W = np.eye(n)
            for i in range(n):
                neighbours = np.where(adj_matrix[i] == 1)[0]
                for j in neighbours:
                    if i < j:
                        W[i, j] = 0.5
                        W[j, i] = 0.5
                        W[i, i] -= 0.5
                        W[j, j] -= 0.5
        else:
            W = ConsensusWeights.create_doubly_stochastic(adj_matrix)
        return W