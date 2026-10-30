import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
import time

class EmbeddingClipping:
    @staticmethod
    def clip_embeddings(U: np.ndarray, V: np.ndarray, R_max: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        clip_norm = np.sqrt(R_max)
        U_clipped = U.copy()
        for i in range(U.shape[0]):
            norm = np.linalg.norm(U[i])
            if norm > clip_norm:
                U_clipped[i] = U_clipped[i] * (clip_norm / norm)
            U_clipped[i] = np.maximum(U_clipped[i], 0)
        
        V_clipped = V.copy()
        for j in range(V.shape[0]):
            norm = np.linalg.norm(V[j])
            if norm > clip_norm:
                V_clipped[j] = V_clipped[j] * (clip_norm / norm)
            V_clipped[j] = np.maximum(V_clipped[j], 0)
        
        return U_clipped, V_clipped
    
    @staticmethod
    def is_valid(U: np.ndarray, V: np.ndarray, R_max: float = 5.0) -> bool:
        clip_norm = np.sqrt(R_max) + 1e-6
        
        for i in range(U.shape[0]):
            if np.any(U[i] < -1e-6): 
                return False
            if np.linalg.norm(U[i]) > clip_norm:
                return False
        
        for j in range(V.shape[0]):
            if np.any(V[j] < -1e-6):
                return False
            if np.linalg.norm(V[j]) > clip_norm:
                return False
        
        return True

class GossipNode:
    def __init__(self, node_id: int, user_ratings: np.ndarray, n_items: int, rank: int, 
                 R_min: float = 0.5, R_max: float = 5.0, use_clipping: bool = True):
        self.node_id = node_id
        self.user_ratings = user_ratings
        self.n_items = n_items
        self.rank = rank
        self.R_min = R_min
        self.R_max = R_max
        self.use_clipping = use_clipping
        
        self.u = None
        self.b = None
        self.Y = None
        self.c = None
        self.t = None
        
        self.initialize_embeddings()
        self.iteration = 0
        self.received_models = {}
        
    def initialize_embeddings(self):
        scale = np.sqrt((self.R_max - self.R_min) / self.rank)
        
        self.u = np.random.uniform(0, scale, (1, self.rank))
        self.b = np.array([self.R_min / 2])
        
        self.Y = np.random.uniform(0, scale, (self.n_items, self.rank))
        self.c = np.full(self.n_items, self.R_min / 2)
        self.t = np.zeros(self.n_items, dtype=int)
        
        if self.use_clipping:
            self.u, self.Y = EmbeddingClipping.clip_embeddings(self.u, self.Y, self.R_max)
    
    def predict(self, item_idx: int) -> float:
        if self.use_clipping:
            self.u, self.Y = EmbeddingClipping.clip_embeddings(self.u, self.Y, self.R_max)
        
        prediction = float(self.u @ self.Y[item_idx].T) + self.b + self.c[item_idx]
        
        prediction = np.clip(prediction, self.R_min, self.R_max)
        
        return prediction
    
    def compute_loss(self, use_mask: bool = True) -> float:
        user_indices, item_indices = np.where(~np.isnan(self.user_ratings))
        
        if len(user_indices) == 0:
            return 0.0
        
        loss = 0.0
        count = 0
        
        for user_idx, item_idx in zip(user_indices, item_indices):
            if user_idx == 0:
                rating = self.user_ratings[user_idx, item_idx]
                prediction = self.predict(item_idx)
                loss += (rating - prediction) ** 2
                count += 1
        
        return loss / max(count, 1)
    
    def local_update(self, n_epochs: int = 1, learning_rate: float = 0.01, 
                    lambda_reg: float = 0.1):
        user_indices, item_indices = np.where(~np.isnan(self.user_ratings))
        
        if len(user_indices) == 0:
            return
        
        for _ in range(n_epochs):
            indices = list(zip(user_indices, item_indices))
            random.shuffle(indices)
            
            for user_idx, item_idx in indices:
                if user_idx == 0:
                    rating = self.user_ratings[user_idx, item_idx]
                    prediction = self.predict(item_idx)
                    error = rating - prediction
                    
                    grad_u = -2 * error * self.Y[item_idx] + lambda_reg * self.u
                    grad_b = -2 * error + lambda_reg * self.b
                    grad_y = (-2 * error * self.u + lambda_reg * self.Y[item_idx].reshape(1, -1)).flatten()
                    grad_c = -2 * error + lambda_reg * self.c[item_idx]
                    
                    self.u -= learning_rate * grad_u
                    self.b -= learning_rate * grad_b
                    self.Y[item_idx] -= learning_rate * grad_y.reshape(-1)
                    self.c[item_idx] -= learning_rate * grad_c
                    
                    self.t[item_idx] += 1
            
            if self.use_clipping:
                self.u, self.Y = EmbeddingClipping.clip_embeddings(self.u, self.Y, self.R_max)
                
            learning_rate = learning_rate * 0.98
    
    def compress(self, compression_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = max(1, int(self.n_items * compression_ratio))
        
        updated_indices = np.where(self.t > 0)[0]
        non_updated_indices = np.where(self.t == 0)[0]
        
        n_updated = min(len(updated_indices), s)
        n_non_updated = max(0, s - n_updated)
        
        selected_indices = []
        
        if n_updated > 0:
            selected_updated = np.random.choice(updated_indices, n_updated, replace=False)
            selected_indices.extend(selected_updated)
        
        if n_non_updated > 0 and len(non_updated_indices) > 0:
            n_to_sample = min(n_non_updated, len(non_updated_indices))
            selected_non_updated = np.random.choice(non_updated_indices, n_to_sample, replace=False)
            selected_indices.extend(selected_non_updated)
        
        if len(selected_indices) < s:
            all_indices = np.arange(self.n_items)
            remaining = np.setdiff1d(all_indices, selected_indices)
            if len(remaining) > 0:
                n_needed = s - len(selected_indices)
                n_to_add = min(n_needed, len(remaining))
                additional = np.random.choice(remaining, n_to_add, replace=False)
                selected_indices.extend(additional)
        
        selected_indices = np.array(selected_indices, dtype=int)
        
        Y_selected = self.Y[selected_indices].copy()
        c_selected = self.c[selected_indices].copy()
        t_selected = self.t[selected_indices].copy()
        
        return Y_selected, c_selected, t_selected, selected_indices
    
    def decompress(self, received_indices: np.ndarray, received_Y: np.ndarray, 
                  received_c: np.ndarray, received_t: np.ndarray):
        for idx, global_idx in enumerate(received_indices):
            if global_idx < self.n_items:
                self.received_models[global_idx] = {
                    'Y': received_Y[idx].copy(),
                    'c': received_c[idx],
                    't': received_t[idx],
                    'node_id': self.node_id 
                }
    
    def merge(self, method: str = 'age_weighted') -> bool:
        if not self.received_models:
            return False
        
        for item_idx, received_model in self.received_models.items():
            Y_rec = received_model['Y']
            c_rec = received_model['c']
            t_rec = received_model['t']
            
            t_self = self.t[item_idx]
            
            if method == 'age_weighted':
                if t_self == 0 and t_rec == 0:
                    continue
                
                if t_self == 0:
                    self.Y[item_idx] = Y_rec
                    self.c[item_idx] = c_rec
                    self.t[item_idx] = t_rec
                elif t_rec == 0:
                    continue
                else:
                    w_self = t_self / (t_self + t_rec)
                    w_rec = t_rec / (t_self + t_rec)
                    
                    self.Y[item_idx] = w_self * self.Y[item_idx] + w_rec * Y_rec
                    self.c[item_idx] = w_self * self.c[item_idx] + w_rec * c_rec
                    self.t[item_idx] = max(t_self, t_rec)
            
            elif method == 'average':
                self.Y[item_idx] = (self.Y[item_idx] + Y_rec) / 2
                self.c[item_idx] = (self.c[item_idx] + c_rec) / 2
                self.t[item_idx] = max(t_self, t_rec)
            
            elif method == 'replace':
                self.Y[item_idx] = Y_rec
                self.c[item_idx] = c_rec
                self.t[item_idx] = t_rec
            
            elif method == 'max_age':
                if t_rec > t_self:
                    self.Y[item_idx] = Y_rec
                    self.c[item_idx] = c_rec
                    self.t[item_idx] = t_rec
        
        if self.use_clipping:
            self.u, self.Y = EmbeddingClipping.clip_embeddings(self.u, self.Y, self.R_max)
        
        self.received_models.clear()
        return True
    
    def get_model_for_sharing(self, compression_ratio: float = 0.1) -> Tuple:
        """Prepare model for sharing with compression"""
        Y_comp, c_comp, t_comp, indices = self.compress(compression_ratio)
        return Y_comp, c_comp, t_comp, indices, self.node_id
    
    def get_validation_loss(self, test_ratings: np.ndarray) -> float:
        if test_ratings is None or np.all(np.isnan(test_ratings)):
            return 0.0
        
        user_indices, item_indices = np.where(~np.isnan(test_ratings))
        loss = 0.0
        count = 0
        
        for user_idx, item_idx in zip(user_indices, item_indices):
            if user_idx == 0:  
                rating = test_ratings[user_idx, item_idx]
                prediction = self.predict(item_idx)
                loss += (rating - prediction) ** 2
                count += 1
        
        return loss / max(count, 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        valid_indices = np.where(~np.isnan(self.user_ratings[0]))[0]
        n_ratings = len(valid_indices)
        
        return {
            'node_id': self.node_id,
            'n_ratings': n_ratings,
            'u_norm': np.linalg.norm(self.u),
            'Y_norm': np.mean(np.linalg.norm(self.Y, axis=1)),
            'avg_age': np.mean(self.t),
            'updated_items': np.sum(self.t > 0),
            'clipping_valid': EmbeddingClipping.is_valid(self.u, self.Y, self.R_max) if self.use_clipping else True
        }

class GossipNetwork:
    def __init__(self, nodes: List[GossipNode], topology: str = 'random', 
                 cycle_time: float = 1728.0, online_probability: float = 1.0,
                 use_clipping: bool = True):
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.topology = topology
        self.cycle_time = cycle_time
        self.online_probability = online_probability
        self.use_clipping = use_clipping
        
        self.online_status = np.ones(self.n_nodes, dtype=bool)
        self.iteration = 0
        self.metrics_history = {
            'loss': [],
            'consensus_error': [],
            'communication_cost': [],
            'online_nodes': [],
            'clipping_violations': []
        }
        
        self._initialize_topology()
    
    def _initialize_topology(self):
        """Initialize network topology"""
        if self.topology == 'complete':
            self.neighbours = {i: [j for j in range(self.n_nodes) if j != i] 
                            for i in range(self.n_nodes)}
        elif self.topology == 'ring':
            self.neighbours = {i: [(i-1) % self.n_nodes, (i+1) % self.n_nodes] 
                            for i in range(self.n_nodes)}
        elif self.topology == 'star':
            self.neighbours = {0: list(range(1, self.n_nodes))}
            for i in range(1, self.n_nodes):
                self.neighbours[i] = [0]
        elif self.topology == 'random':
            self.neighbours = {}
            for i in range(self.n_nodes):
                possible_neighbours = [j for j in range(self.n_nodes) if j != i]
                k = min(3, len(possible_neighbours))
                self.neighbours[i] = random.sample(possible_neighbours, k)
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def select_random_neighbour(self, node_id: int) -> Optional[int]:
        """Select a random online neighbour"""
        if not self.online_status[node_id]:
            return None
        
        if node_id not in self.neighbours:
            return None
        
        available_neighbours = [n for n in self.neighbours[node_id] 
                             if self.online_status[n]]
        
        if not available_neighbours:
            return None
        
        return random.choice(available_neighbours)
    
    def simulate_churn(self):
        """Simulate nodes going online/offline"""
        for i in range(self.n_nodes):
            if random.random() > self.online_probability:
                self.online_status[i] = False
            else:
                self.online_status[i] = True
    
    def check_clipping_violations(self) -> int:
        """Check how many nodes violate embedding constraints"""
        violations = 0
        for node in self.nodes:
            if node.use_clipping:
                if not EmbeddingClipping.is_valid(node.u, node.Y, node.R_max):
                    violations += 1
        return violations
    
    def simulate_cycle(self, compression_ratio: float = 0.1, 
                      merge_method: str = 'age_weighted', 
                      local_epochs: int = 1) -> Dict[str, float]:
        self.iteration += 1
        self.simulate_churn()
        
        total_communication = 0
        successful_exchanges = 0
        
        node_order = list(range(self.n_nodes))
        random.shuffle(node_order)
        
        for node_id in node_order:
            if not self.online_status[node_id]:
                continue
            
            node = self.nodes[node_id]
            neighbour_id = self.select_random_neighbour(node_id)
            
            if neighbour_id is None or not self.online_status[neighbour_id]:
                continue
            
            neighbour_node = self.nodes[neighbour_id]
            
            Y_send, c_send, t_send, indices_send, _ = node.get_model_for_sharing(compression_ratio)
            
            comm_cost = (Y_send.size * 8 +  
                        c_send.size * 8 + 
                        t_send.size * 4 +  
                        indices_send.size * 4)
            total_communication += comm_cost
            
            neighbour_node.decompress(indices_send, Y_send, c_send, t_send)
            
            if random.random() < 0.5:
                Y_rec, c_rec, t_rec, indices_rec, _ = neighbour_node.get_model_for_sharing(compression_ratio)
                node.decompress(indices_rec, Y_rec, c_rec, t_rec)
                
                comm_cost_rec = (Y_rec.size * 8 + c_rec.size * 8 + 
                               t_rec.size * 4 + indices_rec.size * 4)
                total_communication += comm_cost_rec
            
            successful_exchanges += 1
        
        clipping_violations = 0
        for node in self.nodes:
            if self.online_status[node.node_id]:
                if random.random() < 0.7:
                    node.merge(method=merge_method)
                
                node.local_update(n_epochs=local_epochs)
                
                if node.use_clipping and not EmbeddingClipping.is_valid(node.u, node.Y, node.R_max):
                    clipping_violations += 1
                    node.u, node.Y = EmbeddingClipping.clip_embeddings(node.u, node.Y, node.R_max)
        
        metrics = self.get_global_metrics()
        metrics['communication_cost'] = total_communication
        metrics['successful_exchanges'] = successful_exchanges
        metrics['online_nodes'] = np.sum(self.online_status)
        metrics['clipping_violations'] = clipping_violations
        
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics
    
    def get_global_metrics(self) -> Dict[str, float]:
        total_loss = 0.0
        consensus_error = 0.0
        n_active = 0
        
        all_Y = []
        active_nodes = []
        
        for node in self.nodes:
            if self.online_status[node.node_id]:
                loss = node.compute_loss()
                total_loss += loss
                all_Y.append(node.Y)
                active_nodes.append(node)
                n_active += 1
        
        if n_active > 0:
            avg_loss = total_loss / n_active
            
            if len(all_Y) > 0:
                Y_mean = np.mean(all_Y, axis=0)
                for Y in all_Y:
                    consensus_error += np.mean((Y - Y_mean) ** 2)
                consensus_error /= len(all_Y)
        else:
            avg_loss = 0.0
            consensus_error = 0.0
        
        return {
            'loss': avg_loss,
            'consensus_error': np.sqrt(consensus_error),  
            'iteration': self.iteration,
            'active_nodes': n_active
        }
    
    def run_experiment(self, n_cycles: int, track_metrics: bool = True, 
                      compression_ratio: float = 0.1, local_epochs: int = 1,
                      merge_method: str = 'age_weighted') -> Dict[str, Any]:
        print(f"Starting Gossip Learning Experiment")
        print(f"Nodes: {self.n_nodes}, Cycles: {n_cycles}")
        print(f"Compression: {compression_ratio:.0%}, Clipping: {self.use_clipping}")
        print("=" * 60)
        
        all_metrics = []
        
        for cycle in range(n_cycles):
            metrics = self.simulate_cycle(
                compression_ratio=compression_ratio,
                merge_method=merge_method,
                local_epochs=local_epochs
            )
            
            all_metrics.append(metrics)
            
            print(f"Cycle {cycle:4d} | Loss: {float(metrics['loss']):.4f} | "
                    f"Consensus: {metrics['consensus_error']:.4f} | "
                    f"Online: {metrics['online_nodes']}/{self.n_nodes}")
        
        print("=" * 60)
        
        final_metrics = self.get_global_metrics()
        print(f"Final Loss: {float(final_metrics['loss']):.4f}")
        print(f"Final Consensus Error: {final_metrics['consensus_error']:.4f}")
        print(f"Total Communication: {sum(self.metrics_history['communication_cost']):.0f} bytes")
        
        if self.use_clipping:
            violations = self.check_clipping_violations()
            print(f"Clipping Violations: {violations}/{self.n_nodes} nodes")
        
        return {
            'final_metrics': final_metrics,
            'all_metrics': all_metrics,
            'history': self.metrics_history,
            'node_stats': [node.get_stats() for node in self.nodes]
        }
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics for all nodes"""
        stats = {
            'nodes': [],
            'summary': {
                'avg_ratings_per_node': 0,
                'avg_u_norm': 0,
                'avg_Y_norm': 0,
                'avg_age': 0,
                'online_nodes': int(np.sum(self.online_status))
            }
        }
        
        total_ratings = 0
        total_u_norm = 0
        total_Y_norm = 0
        total_age = 0
        
        for node in self.nodes:
            node_stats = node.get_stats()
            stats['nodes'].append(node_stats)
            
            total_ratings += node_stats['n_ratings']
            total_u_norm += node_stats['u_norm']
            total_Y_norm += node_stats['Y_norm']
            total_age += node_stats['avg_age']
        
        n_active = len([n for n in self.nodes if self.online_status[n.node_id]])
        if n_active > 0:
            stats['summary']['avg_ratings_per_node'] = total_ratings / n_active
            stats['summary']['avg_u_norm'] = total_u_norm / n_active
            stats['summary']['avg_Y_norm'] = total_Y_norm / n_active
            stats['summary']['avg_age'] = total_age / n_active
        
        return stats