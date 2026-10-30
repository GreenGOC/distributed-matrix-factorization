import numpy as np
import random
from typing import Dict, List, Tuple, Any

class AttackSimulator:
    @staticmethod
    def simulate_data_poisoning(ratings: np.ndarray, attack_type: str = 'random', 
                               attack_params: Dict[str, Any] = None) -> np.ndarray:
        if attack_params is None:
            attack_params = {}
        
        poisoned_ratings = ratings.copy()
        n_users, n_items = ratings.shape
        
        if attack_type == 'random':
            malicious_ratio = attack_params.get('malicious_ratio', 0.2)
            attack_strength = attack_params.get('attack_strength', 4.0)
            
            n_malicious = max(1, int(n_users * malicious_ratio))
            malicious_users = random.sample(range(n_users), n_malicious)
            
            for user_idx in malicious_users:
                valid_mask = ~np.isnan(ratings[user_idx])
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    n_to_corrupt = max(1, int(len(valid_indices) * 0.3))
                    indices_to_corrupt = random.sample(list(valid_indices), 
                                                     min(n_to_corrupt, len(valid_indices)))
                    
                    for item_idx in indices_to_corrupt:
                        original_rating = ratings[user_idx, item_idx]
                        if attack_strength > 0:
                            extreme_value = random.choice([0.5, 5.0])
                            poisoned_ratings[user_idx, item_idx] = extreme_value
                        else:
                            random_rating = np.random.uniform(0.5, 5.0)
                            poisoned_ratings[user_idx, item_idx] = random_rating
        
        elif attack_type == 'average':
            target_item = attack_params.get('target_item', 0)
            target_rating = attack_params.get('target_rating', 5.0)
            malicious_ratio = attack_params.get('malicious_ratio', 0.2)
            
            n_malicious = max(1, int(n_users * malicious_ratio))
            malicious_users = random.sample(range(n_users), n_malicious)
            
            for user_idx in malicious_users:
                if target_item < n_items:
                    poisoned_ratings[user_idx, target_item] = target_rating
        
        elif attack_type == 'nuke':
            malicious_ratio = attack_params.get('malicious_ratio', 0.3)
            n_malicious = max(1, int(n_users * malicious_ratio))
            malicious_users = random.sample(range(n_users), n_malicious)
            
            for user_idx in malicious_users:
                for item_idx in range(n_items):
                    if not np.isnan(ratings[user_idx, item_idx]):
                        poisoned_ratings[user_idx, item_idx] = 0.5
        
        elif attack_type == 'hybrid':
            push_item = attack_params.get('push_item', 0)
            nuke_items = attack_params.get('nuke_items', [])
            malicious_ratio = attack_params.get('malicious_ratio', 0.25)
            
            n_malicious = max(1, int(n_users * malicious_ratio))
            malicious_users = random.sample(range(n_users), n_malicious)
            
            for user_idx in malicious_users:
                if push_item < n_items:
                    poisoned_ratings[user_idx, push_item] = 5.0
                
                for item_idx in nuke_items:
                    if item_idx < n_items and not np.isnan(ratings[user_idx, item_idx]):
                        poisoned_ratings[user_idx, item_idx] = 0.5
        
        return poisoned_ratings

    @staticmethod
    def simulate_model_attack(embeddings: np.ndarray, attack_type: str = 'gradient',
                            attack_params: Dict[str, Any] = None) -> np.ndarray:
        if attack_params is None:
            attack_params = {}
        
        manipulated_embeddings = embeddings.copy()
        
        if attack_type == 'gradient':
            manipulation_scale = attack_params.get('manipulation_scale', 2.0)
            noise = np.random.randn(*embeddings.shape) * manipulation_scale
            manipulated_embeddings += noise
        
        elif attack_type == 'reverse':
            manipulation_scale = attack_params.get('manipulation_scale', -1.0)
            manipulated_embeddings *= manipulation_scale
        
        elif attack_type == 'randomize':
            scale = attack_params.get('scale', 3.0)
            manipulated_embeddings = np.random.randn(*embeddings.shape) * scale
        
        elif attack_type == 'zero':
            zero_ratio = attack_params.get('zero_ratio', 0.5)
            n_zero = int(embeddings.size * zero_ratio)
            zero_indices = random.sample(range(embeddings.size), n_zero)
            manipulated_embeddings.flat[zero_indices] = 0.0
        
        return manipulated_embeddings

    @staticmethod
    def simulate_byzantine_attack(node, attack_type: str = 'sign_flip',
                                 attack_params: Dict[str, Any] = None):
        if attack_params is None:
            attack_params = {}
        
        if attack_type == 'sign_flip':
            flip_prob = attack_params.get('flip_prob', 0.3)
            mask = np.random.random(node.u.shape) < flip_prob
            node.u[mask] *= -1
        
        elif attack_type == 'amplify':
            amplify_factor = attack_params.get('amplify_factor', 3.0)
            node.u *= amplify_factor
        
        elif attack_type == 'random_gaussian':
            std_dev = attack_params.get('std_dev', 5.0)
            node.u = np.random.randn(*node.u.shape) * std_dev
        
        elif attack_type == 'same_value':
            constant_value = attack_params.get('constant_value', 10.0)
            node.u.fill(constant_value)
        
        if node.use_clipping:
            node.u, node.Y = AttackSimulator.clip_embeddings_for_attack(node.u, node.Y, node.R_max)
    
    @staticmethod
    def clip_embeddings_for_attack(U: np.ndarray, V: np.ndarray, R_max: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        clip_norm = np.sqrt(R_max)
        U_clipped = U.copy()
        for i in range(U.shape[0]):
            norm = np.linalg.norm(U[i])
            if norm > clip_norm:
                U_clipped[i] = U_clipped[i] * (clip_norm / norm)
        
        V_clipped = V.copy()
        for j in range(V.shape[0]):
            norm = np.linalg.norm(V[j])
            if norm > clip_norm:
                V_clipped[j] = V_clipped[j] * (clip_norm / norm)
        
        return U_clipped, V_clipped

    @staticmethod
    def create_malicious_nodes(nodes: List, attack_type: str = 'data_poisoning',
                              attack_params: Dict[str, Any] = None) -> List:
        if attack_params is None:
            attack_params = {}
        
        malicious_ratio = attack_params.get('malicious_ratio', 0.2)
        n_nodes = len(nodes)
        n_malicious = max(1, int(n_nodes * malicious_ratio))
        malicious_indices = random.sample(range(n_nodes), n_malicious)
        
        if attack_type == 'data_poisoning':
            attack_strength = attack_params.get('attack_strength', 4.0)
            for idx in malicious_indices:
                node = nodes[idx]
                valid_mask = ~np.isnan(node.user_ratings[0])
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    n_to_corrupt = max(1, int(len(valid_indices) * 0.3))
                    indices_to_corrupt = random.sample(list(valid_indices), 
                                                     min(n_to_corrupt, len(valid_indices)))
                    
                    for item_idx in indices_to_corrupt:
                        if attack_strength > 0:
                            node.user_ratings[0, item_idx] = random.choice([0.5, 5.0])
                        else:
                            node.user_ratings[0, item_idx] = np.random.uniform(0.5, 5.0)
        
        elif attack_type == 'model_manipulation':
            manipulation_scale = attack_params.get('manipulation_scale', 2.0)
            for idx in malicious_indices:
                node = nodes[idx]
                noise = np.random.randn(*node.u.shape) * manipulation_scale
                node.u += noise
        
        return nodes

    @staticmethod
    def measure_attack_impact(original_nodes: List, attacked_nodes: List) -> Dict[str, float]:
        if len(original_nodes) != len(attacked_nodes):
            return {}
        
        n_nodes = len(original_nodes)
        u_diff = 0.0
        Y_diff = 0.0
        predictions_diff = 0.0
        
        for i in range(n_nodes):
            orig = original_nodes[i]
            attacked = attacked_nodes[i]
            
            u_diff += np.linalg.norm(orig.u - attacked.u)
            Y_diff += np.linalg.norm(orig.Y - attacked.Y)
            
            sample_items = min(10, orig.n_items)
            test_indices = random.sample(range(orig.n_items), sample_items)
            
            for item_idx in test_indices:
                pred_orig = orig.predict(item_idx)
                pred_attacked = attacked.predict(item_idx)
                predictions_diff += abs(pred_orig - pred_attacked)
        
        return {
            'avg_u_diff': u_diff / n_nodes,
            'avg_Y_diff': Y_diff / n_nodes,
            'avg_prediction_diff': predictions_diff / (n_nodes * sample_items) if sample_items > 0 else 0.0
        }