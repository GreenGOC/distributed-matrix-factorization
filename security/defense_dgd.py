import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

class RLTrustDefenseDGD:
    def __init__(self, n_nodes, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        self.n_nodes = n_nodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.trust_scores = np.ones(n_nodes) * 0.5
        self.update_history = {i: [] for i in range(n_nodes)}
        self.attack_detections = {i: 0 for i in range(n_nodes)}
        self.false_positives = {i: 0 for i in range(n_nodes)}
        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.rewards_history = []
        self.trust_history = []
    
    def extract_state_features(self, node_id, update_vector, previous_vector=None):
        current_trust = self.trust_scores[node_id]
        update_norm = np.linalg.norm(update_vector)
        update_std = np.std(update_vector) if update_vector.size > 1 else 0
        
        if previous_vector is not None and previous_vector.shape == update_vector.shape:
            similarity = np.corrcoef(update_vector.flatten(), previous_vector.flatten())[0, 1]
            if np.isnan(similarity):
                similarity = 0
        else:
            similarity = 0
        
        trust_level = discretize_value(current_trust, bins=[0, 0.3, 0.7, 1.0])
        norm_level = discretize_value(update_norm, bins=[0, 1, 5, 10, np.inf])
        similarity_level = discretize_value(similarity, bins=[-1, -0.5, 0, 0.5, 1])
        
        return (node_id, trust_level, norm_level, similarity_level)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)
    
    def calculate_reward(self, node_id, update_vector, global_consensus, is_attack_detected):
        base_reward = 0
        
        if global_consensus is not None and update_vector.shape == global_consensus.shape:
            update_normalized = update_vector / (np.linalg.norm(update_vector) + 1e-8)
            consensus_normalized = global_consensus / (np.linalg.norm(global_consensus) + 1e-8)
            similarity = np.dot(update_normalized.flatten(), consensus_normalized.flatten())
            similarity_reward = max(-1, min(1, similarity))
            base_reward += similarity_reward * 0.5
        
        if is_attack_detected:
            base_reward -= 2.0
            self.attack_detections[node_id] += 1
        else:
            base_reward += 0.1
        
        if len(self.update_history[node_id]) >= 2:
            last_update = self.update_history[node_id][-1]
            if last_update.shape == update_vector.shape:
                consistency = np.corrcoef(last_update.flatten(), update_vector.flatten())[0, 1]
                if not np.isnan(consistency):
                    base_reward += consistency * 0.3
        
        return base_reward
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
        self.last_reward = reward
        self.rewards_history.append(reward)
    
    def update_trust_score(self, node_id, reward, action):
        trust_change = 0
        
        if action == 2:
            if reward > 0:
                trust_change = min(0.1, reward * 0.2)
            else:
                trust_change = max(-0.2, reward * 0.1)
        elif action == 1:
            trust_change = reward * 0.05
        else:
            trust_change = reward * 0.02
        
        self.trust_scores[node_id] += trust_change
        self.trust_scores[node_id] = np.clip(self.trust_scores[node_id], 0, 1)
        self.trust_history.append((node_id, self.trust_scores[node_id], time.time()))
    
    def detect_anomalies(self, node_id, update_vector, global_stats):
        anomalies = {
            'is_anomalous': False,
            'reasons': [],
            'confidence': 0.0
        }
        
        if update_vector is None or update_vector.size == 0:
            return anomalies
        
        update_norm = np.linalg.norm(update_vector)
        if 'mean_norm' in global_stats and 'std_norm' in global_stats:
            mean_norm = global_stats['mean_norm']
            std_norm = global_stats['std_norm']
            if std_norm > 0:
                z_score_norm = (update_norm - mean_norm) / std_norm
                if abs(z_score_norm) > 3:
                    anomalies['is_anomalous'] = True
                    anomalies['reasons'].append(f"Norm anomaly (z={z_score_norm:.2f})")
                    anomalies['confidence'] += 0.3
        
        if len(self.update_history[node_id]) >= 2:
            last_update = self.update_history[node_id][-1]
            if last_update.shape == update_vector.shape:
                correlation = np.corrcoef(last_update.flatten(), update_vector.flatten())[0, 1]
                if correlation < -0.8:
                    anomalies['is_anomalous'] = True
                    anomalies['reasons'].append(f"Sign flip detected (corr={correlation:.2f})")
                    anomalies['confidence'] += 0.4
        
        anomalies['confidence'] = min(1.0, anomalies['confidence'])
        return anomalies
    
    def filter_updates(self, updates, global_consensus=None, global_stats=None):
        if global_stats is None:
            global_stats = self.compute_global_stats(updates)
        
        filtered_updates = {}
        trusted_indices = []
        action_log = []
        
        for node_id, update in updates.items():
            if update is None or update.size == 0:
                continue
            
            previous_update = None
            if self.update_history[node_id]:
                previous_update = self.update_history[node_id][-1]
            
            state = self.extract_state_features(node_id, update, previous_update)
            self.current_state = state
            
            action = self.choose_action(state)
            self.last_action = action
            
            anomalies = self.detect_anomalies(node_id, update, global_stats)
            is_attack_detected = anomalies['is_anomalous']
            
            reward = self.calculate_reward(node_id, update, global_consensus, is_attack_detected)
            
            next_state = self.simulate_next_state(state, action, reward)
            self.update_q_value(state, action, reward, next_state)
            
            self.update_trust_score(node_id, reward, action)
            
            self.update_history[node_id].append(update.copy())
            
            if action == 0:
                if is_attack_detected:
                    self.attack_detections[node_id] += 1
                else:
                    self.false_positives[node_id] += 1
            elif action == 1:
                weight = 0.3 * self.trust_scores[node_id]
                filtered_updates[node_id] = update * weight
                trusted_indices.append(node_id)
            elif action == 2:
                weight = 0.8 * self.trust_scores[node_id]
                filtered_updates[node_id] = update * weight
                trusted_indices.append(node_id)
            
            action_log.append({
                'node_id': node_id,
                'action': action,
                'reward': reward,
                'trust_score': self.trust_scores[node_id],
                'anomalies': anomalies
            })
        
        return filtered_updates, trusted_indices, action_log
    
    def simulate_next_state(self, current_state, action, reward):
        node_id, trust_level, norm_level, similarity_level = current_state
        
        if reward > 0:
            new_trust_level = min(trust_level + 1, 3)
        else:
            new_trust_level = max(trust_level - 1, 0)
        
        new_norm_level = norm_level
        new_similarity_level = similarity_level
        
        return (node_id, new_trust_level, new_norm_level, new_similarity_level)
    
    def compute_global_stats(self, updates):
        stats = {
            'mean_norm': 0,
            'std_norm': 0,
            'mean_values': 0,
            'std_values': 0,
            'n_updates': len(updates)
        }
        
        if not updates:
            return stats
        
        norms = []
        values = []
        
        for update in updates.values():
            if update is not None and update.size > 0:
                norms.append(np.linalg.norm(update))
                values.extend(update.flatten())
        
        if norms:
            stats['mean_norm'] = np.mean(norms)
            stats['std_norm'] = np.std(norms)
        
        if values:
            stats['mean_values'] = np.mean(values)
            stats['std_values'] = np.std(values)
        
        return stats
    
    def get_trust_distribution(self):
        trust_scores = self.trust_scores
        return {
            'mean_trust': np.mean(trust_scores),
            'std_trust': np.std(trust_scores),
            'min_trust': np.min(trust_scores),
            'max_trust': np.max(trust_scores),
            'trust_histogram': np.histogram(trust_scores, bins=10)[0].tolist(),
            'high_trust_nodes': np.sum(trust_scores > 0.7),
            'low_trust_nodes': np.sum(trust_scores < 0.3),
            'attack_detections': sum(self.attack_detections.values()),
            'false_positives': sum(self.false_positives.values())
        }
    
    def defend(self, updates, global_consensus=None):
        filtered_updates, trusted_indices, action_log = self.filter_updates(updates, global_consensus)
        return filtered_updates

class AnomalyDetectionDefenseDGD:
    def __init__(self, threshold=2.0, window_size=10):
        self.threshold = threshold
        self.window_size = window_size
        self.update_history = defaultdict(list)
        self.anomaly_counts = defaultdict(int)
    
    def detect_anomalous_updates(self, updates):
        anomalous_nodes = []
        
        if not updates:
            return anomalous_nodes
        
        all_updates = list(updates.values())
        global_norms = [np.linalg.norm(u) for u in all_updates if u is not None]
        
        if len(global_norms) < 2:
            return anomalous_nodes
        
        mean_norm = np.mean(global_norms)
        std_norm = np.std(global_norms)
        
        for node_id, update in updates.items():
            if update is None:
                continue
            
            update_norm = np.linalg.norm(update)
            if std_norm > 0:
                z_score = abs((update_norm - mean_norm) / std_norm)
                if z_score > self.threshold:
                    anomalous_nodes.append(node_id)
                    self.anomaly_counts[node_id] += 1
                    continue
            
            if len(self.update_history[node_id]) >= 2:
                last_update = self.update_history[node_id][-1]
                if last_update.shape == update.shape:
                    correlation = np.corrcoef(last_update.flatten(), update.flatten())[0, 1]
                    if correlation < -0.8:
                        anomalous_nodes.append(node_id)
                        self.anomaly_counts[node_id] += 1
                        continue
            
            self.update_history[node_id].append(update.copy())
            
            if len(self.update_history[node_id]) > self.window_size:
                self.update_history[node_id] = self.update_history[node_id][-self.window_size:]
        
        return anomalous_nodes
    
    def compute_robust_mean(self, updates, method='trimmed'):
        if not updates:
            return None
        
        update_list = list(updates.values())
        
        if method == 'trimmed':
            norms = [np.linalg.norm(u) for u in update_list]
            q1 = np.percentile(norms, 25)
            q3 = np.percentile(norms, 75)
            iqr = q3 - q1
            
            filtered_updates = []
            for u, norm in zip(update_list, norms):
                if q1 - 1.5 * iqr <= norm <= q3 + 1.5 * iqr:
                    filtered_updates.append(u)
            
            if filtered_updates:
                return np.mean(filtered_updates, axis=0)
            else:
                return np.mean(update_list, axis=0)
        
        elif method == 'median':
            return np.median(update_list, axis=0)
        
        else:
            return np.mean(update_list, axis=0)
    
    def filter_outliers(self, updates, threshold=3.0):
        if not updates:
            return {}, []
        
        filtered_updates = {}
        kept_indices = []
        
        update_list = list(updates.values())
        node_ids = list(updates.keys())
        
        norms = [np.linalg.norm(u) for u in update_list]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        for node_id, update, norm in zip(node_ids, update_list, norms):
            if std_norm > 0:
                z_score = abs((norm - mean_norm) / std_norm)
                if z_score <= threshold:
                    filtered_updates[node_id] = update
                    kept_indices.append(node_id)
            else:
                filtered_updates[node_id] = update
                kept_indices.append(node_id)
        
        return filtered_updates, kept_indices
    
    def defend(self, updates, global_consensus=None):
        anomalous_nodes = self.detect_anomalous_updates(updates)
        filtered_updates = {}
        
        for node_id, update in updates.items():
            if node_id not in anomalous_nodes:
                filtered_updates[node_id] = update
        
        return filtered_updates

def discretize_value(value, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 2

import time