import numpy as np
import random
from typing import Dict, List, Tuple, Any

class AttackSimulatorDGD:
    @staticmethod
    def simulate_data_poisoning(nodes, attack_type='random', attack_params=None):
        if attack_params is None:
            attack_params = {}
        
        malicious_ratio = attack_params.get('malicious_ratio', 0.2)
        attack_strength = attack_params.get('attack_strength', 2.0)
        n_nodes = len(nodes)
        n_malicious = max(1, int(n_nodes * malicious_ratio))
        malicious_indices = random.sample(range(n_nodes), n_malicious)
        
        for i, node in enumerate(nodes):
            if i in malicious_indices:
                if attack_type == 'random':
                    noise_mask = np.random.random(node.Yj.shape) < 0.3
                    noise = np.random.randn(*node.Yj.shape) * attack_strength
                    node.Yj = np.where(noise_mask, np.clip(node.Yj + noise, 0.5, 5.0), node.Yj)
                
                elif attack_type == 'reverse':
                    node.Yj = 5.0 - node.Yj
                    node.Yj = np.clip(node.Yj, 0.5, 5.0)
                
                elif attack_type == 'constant':
                    node.Yj = np.full_like(node.Yj, 2.5)
        
        return nodes
    
    @staticmethod
    def simulate_model_manipulation(nodes, attack_type='gradient', attack_params=None):
        if attack_params is None:
            attack_params = {}
        
        malicious_ratio = attack_params.get('malicious_ratio', 0.2)
        manipulation_scale = attack_params.get('manipulation_scale', 2.0)
        n_nodes = len(nodes)
        n_malicious = max(1, int(n_nodes * malicious_ratio))
        malicious_indices = random.sample(range(n_nodes), n_malicious)
        
        for i, node in enumerate(nodes):
            if i in malicious_indices:
                if attack_type == 'gradient':
                    noise = np.random.randn(*node.Uj.shape) * manipulation_scale
                    node.Uj += noise
                    noise_v = np.random.randn(*node.Vj.shape) * manipulation_scale
                    node.Vj += noise_v
                
                elif attack_type == 'reverse':
                    node.Uj = node.Uj * -1.0
                    node.Vj = node.Vj * -1.0
                
                elif attack_type == 'zero':
                    node.Uj = np.zeros_like(node.Uj)
                    node.Vj = np.zeros_like(node.Vj)
                
                elif attack_type == 'amplify':
                    node.Uj = node.Uj * manipulation_scale
                    node.Vj = node.Vj * manipulation_scale
        
        return nodes
    
    @staticmethod
    def simulate_byzantine_attack(nodes, attack_type='sign_flip', attack_params=None):
        if attack_params is None:
            attack_params = {}
        
        malicious_ratio = attack_params.get('malicious_ratio', 0.2)
        flip_prob = attack_params.get('flip_prob', 0.3)
        n_nodes = len(nodes)
        n_malicious = max(1, int(n_nodes * malicious_ratio))
        malicious_indices = random.sample(range(n_nodes), n_malicious)
        
        for i, node in enumerate(nodes):
            if i in malicious_indices:
                if attack_type == 'sign_flip':
                    mask_u = np.random.random(node.Uj.shape) < flip_prob
                    node.Uj[mask_u] *= -1.0
                    mask_v = np.random.random(node.Vj.shape) < flip_prob
                    node.Vj[mask_v] *= -1.0
                
                elif attack_type == 'gaussian':
                    node.Uj = np.random.randn(*node.Uj.shape) * 5.0
                    node.Vj = np.random.randn(*node.Vj.shape) * 5.0
                
                elif attack_type == 'same_value':
                    node.Uj = np.full_like(node.Uj, 10.0)
                    node.Vj = np.full_like(node.Vj, 10.0)
        
        return nodes
    
    @staticmethod
    def create_malicious_nodes(nodes, attack_type='data_poisoning', attack_params=None):
        if attack_params is None:
            attack_params = {}
        
        if attack_type == 'data_poisoning':
            return AttackSimulatorDGD.simulate_data_poisoning(nodes, 'random', attack_params)
        elif attack_type == 'model_manipulation':
            return AttackSimulatorDGD.simulate_model_manipulation(nodes, 'gradient', attack_params)
        elif attack_type == 'byzantine':
            return AttackSimulatorDGD.simulate_byzantine_attack(nodes, 'sign_flip', attack_params)
        else:
            return nodes
    
    @staticmethod
    def measure_attack_impact(original_nodes, attacked_nodes):
        if len(original_nodes) != len(attacked_nodes):
            return {}
        
        n_nodes = len(original_nodes)
        U_diff = 0.0
        V_diff = 0.0
        Y_diff = 0.0
        
        for i in range(n_nodes):
            orig = original_nodes[i]
            attacked = attacked_nodes[i]
            
            U_diff += np.linalg.norm(orig.Uj - attacked.Uj)
            V_diff += np.linalg.norm(orig.Vj - attacked.Vj)
            
            if orig.Yj.shape == attacked.Yj.shape:
                Y_diff += np.linalg.norm(orig.Yj - attacked.Yj)
        
        return {
            'avg_U_diff': U_diff / n_nodes,
            'avg_V_diff': V_diff / n_nodes,
            'avg_Y_diff': Y_diff / n_nodes,
            'total_impact': (U_diff + V_diff + Y_diff) / (3 * n_nodes)
        }
    
    @staticmethod
    def inject_malicious_updates(nodes, malicious_indices, attack_method='noise'):
        for idx in malicious_indices:
            if idx < len(nodes):
                node = nodes[idx]
                if attack_method == 'noise':
                    noise = np.random.randn(*node.Uj.shape) * 2.0
                    node.Uj += noise
                    node.Vj += np.random.randn(*node.Vj.shape) * 2.0
                elif attack_method == 'sign_flip':
                    node.Uj *= -1.0
                    node.Vj *= -1.0
                elif attack_method == 'zero':
                    node.Uj = np.zeros_like(node.Uj)
                    node.Vj = np.zeros_like(node.Vj)
        
        return nodes
    
    @staticmethod
    def apply_attack_during_training(nodes, iteration, attack_params=None):
        if attack_params is None:
            return nodes
        
        attack_prob = attack_params.get('attack_prob', 0.1)
        attack_strength = attack_params.get('attack_strength', 1.0)
        
        if random.random() < attack_prob:
            n_nodes = len(nodes)
            n_to_attack = max(1, int(n_nodes * 0.2))
            attack_indices = random.sample(range(n_nodes), n_to_attack)
            
            for idx in attack_indices:
                node = nodes[idx]
                noise = np.random.randn(*node.Uj.shape) * attack_strength
                node.Uj += noise
        
        return nodes