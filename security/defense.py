import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import time

class RLTrustDefense:
    def __init__(self, n_nodes: int, learning_rate: float = 0.1, 
                 gamma: float = 0.95, epsilon: float = 0.1):
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
        
    def extract_state_features(self, node_id: int, update: np.ndarray, 
                              previous_update: Optional[np.ndarray] = None) -> Tuple:
        current_trust = self.trust_scores[node_id]
        update_norm = np.linalg.norm(update)
        update_std = np.std(update) if update.size > 1 else 0
        
        if previous_update is not None and previous_update.shape == update.shape:
            similarity = np.corrcoef(update.flatten(), previous_update.flatten())[0, 1]
            if np.isnan(similarity):
                similarity = 0
        else:
            similarity = 0
        
        trust_level = discretize_value(current_trust, bins=[0, 0.3, 0.7, 1.0])
        norm_level = discretize_value(update_norm, bins=[0, 1, 5, 10, np.inf])
        similarity_level = discretize_value(similarity, bins=[-1, -0.5, 0, 0.5, 1])
        
        return (node_id, trust_level, norm_level, similarity_level)
    
    def choose_action(self, state: Tuple) -> int:
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)
    
    def calculate_reward(self, node_id: int, update: np.ndarray, 
                        global_consensus: np.ndarray, 
                        is_attack_detected: bool) -> float:
        base_reward = 0
        
        if global_consensus is not None and update.shape == global_consensus.shape:
            update_normalized = update / (np.linalg.norm(update) + 1e-8)
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
            if last_update.shape == update.shape:
                consistency = np.corrcoef(last_update.flatten(), update.flatten())[0, 1]
                if not np.isnan(consistency):
                    base_reward += consistency * 0.3
        
        return base_reward
    
    def update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        current_q = self.q_table[state][action]
        
        next_max_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        self.last_reward = reward
        self.rewards_history.append(reward)
    
    def update_trust_score(self, node_id: int, reward: float, action: int):
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
    
    def detect_anomalies(self, node_id: int, update: np.ndarray, 
                        global_stats: Dict[str, Any]) -> Dict[str, Any]:

        anomalies = {
            'is_anomalous': False,
            'reasons': [],
            'confidence': 0.0
        }
        
        if update is None or update.size == 0:
            return anomalies
        
        update_norm = np.linalg.norm(update)
        if 'mean_norm' in global_stats and 'std_norm' in global_stats:
            mean_norm = global_stats['mean_norm']
            std_norm = global_stats['std_norm']
            
            if std_norm > 0:
                z_score_norm = (update_norm - mean_norm) / std_norm
                if abs(z_score_norm) > 3:  # 3 sigma rule
                    anomalies['is_anomalous'] = True
                    anomalies['reasons'].append(f"Norm anomaly (z={z_score_norm:.2f})")
                    anomalies['confidence'] += 0.3
        
        update_skew = skewness(update.flatten()) if update.size > 2 else 0
        update_kurtosis = kurtosis(update.flatten()) if update.size > 3 else 0
        
        if abs(update_skew) > 5 or abs(update_kurtosis) > 10:
            anomalies['is_anomalous'] = True
            anomalies['reasons'].append(f"Distribution anomaly (skew={update_skew:.2f}, kurt={update_kurtosis:.2f})")
            anomalies['confidence'] += 0.2
        
        if len(self.update_history[node_id]) >= 2:
            last_update = self.update_history[node_id][-1]
            if last_update.shape == update.shape:
                correlation = np.corrcoef(last_update.flatten(), update.flatten())[0, 1]
                if correlation < -0.8:  
                    anomalies['is_anomalous'] = True
                    anomalies['reasons'].append(f"Sign flip detected (corr={correlation:.2f})")
                    anomalies['confidence'] += 0.4
        
        if update.size > 10:
            q1 = np.percentile(update, 25)
            q3 = np.percentile(update, 75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = np.sum((update < lower_bound) | (update > upper_bound))
            outlier_ratio = outliers / update.size
            
            if outlier_ratio > 0.3:
                anomalies['is_anomalous'] = True
                anomalies['reasons'].append(f"High outlier ratio ({outlier_ratio:.1%})")
                anomalies['confidence'] += 0.2 * outlier_ratio
        
        anomalies['confidence'] = min(1.0, anomalies['confidence'])
        
        return anomalies
    
    def filter_updates(self, updates: Dict[int, np.ndarray], 
                      global_consensus: Optional[np.ndarray] = None,
                      global_stats: Optional[Dict[str, Any]] = None) -> Tuple[Dict[int, np.ndarray], List[int]]:

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
    
    def simulate_next_state(self, current_state: Tuple, action: int, reward: float) -> Tuple:

        node_id, trust_level, norm_level, similarity_level = current_state
        
        if reward > 0:
            new_trust_level = min(trust_level + 1, 3)
        else:
            new_trust_level = max(trust_level - 1, 0)
        
        new_norm_level = norm_level
        new_similarity_level = similarity_level
        
        return (node_id, new_trust_level, new_norm_level, new_similarity_level)
    
    def compute_global_stats(self, updates: Dict[int, np.ndarray]) -> Dict[str, Any]:
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
    
    def get_trust_distribution(self) -> Dict[str, Any]:
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
    
    def reset_for_new_round(self):
        self.current_state = None
        self.last_action = None
        self.last_reward = None
    
    def save_state(self, filepath: str):
        state = {
            'q_table': dict(self.q_table),
            'trust_scores': self.trust_scores.tolist(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'attack_detections': self.attack_detections,
            'false_positives': self.false_positives,
            'rewards_history': self.rewards_history,
            'trust_history': self.trust_history
        }
        np.save(filepath, state, allow_pickle=True)
    
    def load_state(self, filepath: str):
        state = np.load(filepath, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(3), state['q_table'])
        self.trust_scores = np.array(state['trust_scores'])
        self.learning_rate = state['learning_rate']
        self.gamma = state['gamma']
        self.epsilon = state['epsilon']
        self.attack_detections = state['attack_detections']
        self.false_positives = state['false_positives']
        self.rewards_history = state['rewards_history']
        self.trust_history = state['trust_history']

    def defend(self, updates: Dict[int, np.ndarray], global_consensus: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        filtered_updates, trusted_indices, action_log = self.filter_updates(updates, global_consensus)
        return filtered_updates

class AnomalyDetectionDefense:

    def __init__(self, threshold: float = 2.0, window_size: int = 10):
        self.threshold = threshold
        self.window_size = window_size
        self.update_history = defaultdict(list)
        self.anomaly_counts = defaultdict(int)
    
    def detect_anomalous_updates(self, updates: Dict[int, np.ndarray]) -> List[int]:

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
            
            if update.size > 3:
                skew = skewness(update.flatten())
                kurt = kurtosis(update.flatten())
                
                if abs(skew) > 5 or abs(kurt) > 10:
                    anomalous_nodes.append(node_id)
                    self.anomaly_counts[node_id] += 1
                    continue
            
            if len(self.update_history[node_id]) >= 2:
                last_update = self.update_history[node_id][-1]
                if last_update.shape == update.shape:
                    correlation = np.corrcoef(last_update.flatten(), update.flatten())[0, 1]
                    if correlation < -0.8:  # Sign flip detection
                        anomalous_nodes.append(node_id)
                        self.anomaly_counts[node_id] += 1
                        continue
            
            self.update_history[node_id].append(update.copy())
            
            if len(self.update_history[node_id]) > self.window_size:
                self.update_history[node_id] = self.update_history[node_id][-self.window_size:]
        
        return anomalous_nodes
    
    def compute_robust_mean(self, updates: Dict[int, np.ndarray], 
                           method: str = 'trimmed') -> Optional[np.ndarray]:

        if not updates:
            return None
        
        update_list = list(updates.values())
        node_ids = list(updates.keys())
        
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
        
        elif method == 'weighted':
            weights = []
            for node_id, update in updates.items():
                if len(self.update_history[node_id]) >= 2:
                    last_updates = self.update_history[node_id][-2:]
                    if len(last_updates) == 2:
                        consistency = abs(np.corrcoef(last_updates[0].flatten(), 
                                                    last_updates[1].flatten())[0, 1])
                        if np.isnan(consistency):
                            consistency = 0
                        weights.append(consistency)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)
            
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_sum = sum(w * u for w, u in zip(weights, update_list))
                return weighted_sum
            else:
                return np.mean(update_list, axis=0)
        
        else:
            return np.mean(update_list, axis=0)
    
    def filter_outliers(self, updates: Dict[int, np.ndarray], 
                       threshold: float = 3.0) -> Tuple[Dict[int, np.ndarray], List[int]]:

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
    
    def compute_update_statistics(self, updates: Dict[int, np.ndarray]) -> Dict[str, Any]:

        stats = {
            'n_updates': len(updates),
            'mean_norm': 0,
            'std_norm': 0,
            'median_norm': 0,
            'min_norm': 0,
            'max_norm': 0,
            'anomaly_count': 0
        }
        
        if not updates:
            return stats
        
        norms = [np.linalg.norm(u) for u in updates.values() if u is not None]
        
        if norms:
            stats['mean_norm'] = float(np.mean(norms))
            stats['std_norm'] = float(np.std(norms))
            stats['median_norm'] = float(np.median(norms))
            stats['min_norm'] = float(np.min(norms))
            stats['max_norm'] = float(np.max(norms))
            
            anomalous = self.detect_anomalous_updates(updates)
            stats['anomaly_count'] = len(anomalous)
            stats['anomaly_ratio'] = len(anomalous) / len(updates)
        
        return stats

    def defend(self, updates: Dict[int, np.ndarray], global_consensus: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        anomalous_nodes = self.detect_anomalous_updates(updates)
        filtered_updates = {}
        
        for node_id, update in updates.items():
            if node_id not in anomalous_nodes:
                filtered_updates[node_id] = update
        
        return filtered_updates

class AdaptiveDefenseSystem:

    
    def __init__(self, n_nodes: int, use_rl: bool = True, use_statistical: bool = True):
        self.n_nodes = n_nodes
        self.use_rl = use_rl
        self.use_statistical = use_statistical
        
        if use_rl:
            self.rl_defense = RLTrustDefense(n_nodes)
        
        if use_statistical:
            self.statistical_defense = AnomalyDetectionDefense()
        
        self.defense_mode = 'hybrid'  # 'rl', 'statistical', or 'hybrid'
        self.performance_history = []
    
    def defend(self, updates: Dict[int, np.ndarray], 
               global_consensus: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:

        if self.defense_mode == 'rl' and self.use_rl:
            filtered_updates = self.rl_defense.defend(updates, global_consensus)
            defense_info = {
                'mode': 'rl',
                'trusted_count': len(filtered_updates)
            }
        
        elif self.defense_mode == 'statistical' and self.use_statistical:
            filtered_updates = self.statistical_defense.defend(updates, global_consensus)
            defense_info = {
                'mode': 'statistical',
                'trusted_count': len(filtered_updates)
            }
        
        else: 
            anomalous_nodes = []
            if self.use_statistical:
                anomalous_nodes = self.statistical_defense.detect_anomalous_updates(updates)
            
            remaining_updates = {}
            for node_id, update in updates.items():
                if node_id not in anomalous_nodes:
                    remaining_updates[node_id] = update
            
            if self.use_rl and remaining_updates:
                filtered_updates = self.rl_defense.defend(remaining_updates, global_consensus)
            else:
                filtered_updates = remaining_updates
            
            defense_info = {
                'mode': 'hybrid',
                'trusted_count': len(filtered_updates),
                'anomalous_count': len(anomalous_nodes)
            }
        
        self.performance_history.append({
            'timestamp': time.time(),
            'n_updates': len(updates),
            'n_filtered': len(filtered_updates),
            'defense_info': defense_info
        })
        
        return filtered_updates
    
    def adapt_defense_mode(self, attack_detection_rate: float, false_positive_rate: float):

        if attack_detection_rate > 0.8 and false_positive_rate < 0.2:
            self.defense_mode = 'hybrid'
        elif attack_detection_rate > 0.9:
            self.defense_mode = 'rl'
        elif false_positive_rate > 0.3:
            self.defense_mode = 'statistical'
        else:
            self.defense_mode = 'hybrid'
    
    def get_defense_stats(self) -> Dict[str, Any]:

        stats = {
            'defense_mode': self.defense_mode,
            'performance_history_length': len(self.performance_history)
        }
        
        if self.use_rl:
            stats['rl_stats'] = self.rl_defense.get_trust_distribution()
        
        if self.use_statistical and self.performance_history:
            recent = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            stats['recent_performance'] = {
                'avg_filter_rate': np.mean([p['n_filtered'] / max(1, p['n_updates']) for p in recent]),
                'n_defense_actions': len(recent)
            }
        
        return stats

def discretize_value(value: float, bins: List[float]) -> int:

    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 2  # Return last bin

def skewness(data: np.ndarray) -> float:

    if len(data) < 3:
        return 0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    
    skew = np.mean(((data - mean) / std) ** 3)
    return float(skew)

def kurtosis(data: np.ndarray) -> float:
    if len(data) < 4:
        return 0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    
    kurt = np.mean(((data - mean) / std) ** 4) - 3 
    return float(kurt)

class DefenseEvaluator:
    
    @staticmethod
    def evaluate_defense_performance(original_updates: Dict[int, np.ndarray],
                                    filtered_updates: Dict[int, np.ndarray],
                                    malicious_nodes: List[int]) -> Dict[str, float]:
        metrics = {
            'detection_rate': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'filter_rate': 0.0
        }
        
        if not original_updates:
            return metrics
        
        n_total = len(original_updates)
        n_malicious = len(malicious_nodes)
        n_benign = n_total - n_malicious
        
        if n_malicious == 0:
            n_filtered = len(filtered_updates)
            metrics['filter_rate'] = n_filtered / n_total
            
            n_false_positives = n_total - n_filtered
            if n_benign > 0:
                metrics['false_positive_rate'] = n_false_positives / n_benign
            
            return metrics
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for node_id in original_updates.keys():
            is_malicious = node_id in malicious_nodes
            is_filtered = node_id not in filtered_updates
            
            if is_malicious and is_filtered:
                true_positives += 1
            elif not is_malicious and is_filtered:
                false_positives += 1
            elif is_malicious and not is_filtered:
                false_negatives += 1
        
        if n_malicious > 0:
            metrics['detection_rate'] = true_positives / n_malicious
            metrics['false_negative_rate'] = false_negatives / n_malicious
        
        if n_benign > 0:
            metrics['false_positive_rate'] = false_positives / n_benign
        
        if true_positives + false_positives > 0:
            metrics['precision'] = true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives > 0:
            metrics['recall'] = true_positives / (true_positives + false_negatives)
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        metrics['filter_rate'] = (n_total - len(filtered_updates)) / n_total
        
        return metrics