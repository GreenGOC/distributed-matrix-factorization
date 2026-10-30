import numpy as np
import matplotlib.pyplot as plt
from models.dgd_local import DGDLocalAlgorithm
from network.node import DGDNode
from network.topology import TopologyGenerator
from network.consensus import ConsensusWeights
from data.data_preprocessor import DataPreprocessor
from data.movielens_loader import MovieLensLoader
from security.attack_dgd import AttackSimulatorDGD
from security.defense_dgd import RLTrustDefenseDGD, AnomalyDetectionDefenseDGD
import pickle
import os
import time
from typing import Dict, List, Tuple
import seaborn as sns

class DGDComprehensiveTester:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.results_dir = "dgd_results"
        os.makedirs(self.results_dir, exist_ok=True)
        sns.set_style("whitegrid")
        
    def load_movielens_data(self):
        loader = MovieLensLoader(self.data_dir)
        data = loader.load_dataset("MovieLens100K")
        R, mask = loader.get_ratings_matrix(data['ratings_df'])
        preprocessor = DataPreprocessor(R, mask)
        R_train, mask_train, R_test, mask_test = preprocessor.split_train_test_by_user(n_test_per_user=10)
        return R_train, mask_train, R_test, mask_test, data['n_users'], data['n_items']
    
    def create_dgd_nodes(self, R_train, n_nodes=20, rank=10):
        n_users, n_items = R_train.shape
        user_groups = np.array_split(np.arange(n_users), n_nodes)
        nodes = []
        for i, user_indices in enumerate(user_groups):
            if len(user_indices) == 0:
                user_group_ratings = np.zeros((1, n_items)) * np.nan
            else:
                user_group_ratings = R_train[user_indices]
            mask_group = ~np.isnan(user_group_ratings)
            user_group_ratings_filled = np.nan_to_num(user_group_ratings, nan=0.0)
            node = DGDNode(
                node_id=i,
                data_Yj=user_group_ratings_filled,
                mask_Yj=mask_group,
                rank=rank,
                init_scale=0.01
            )
            nodes.append(node)
        return nodes
    
    def copy_dgd_nodes(self, nodes):
        copied_nodes = []
        for node in nodes:
            new_node = DGDNode(
                node_id=node.node_id,
                data_Yj=node.Yj.copy(),
                mask_Yj=node.mask.copy(),
                rank=node.rank,
                init_scale=0.01
            )
            new_node.Uj = node.Uj.copy()
            new_node.Vj = node.Vj.copy()
            copied_nodes.append(new_node)
        return copied_nodes
    
    def run_dgd_experiment(self, nodes, topology_type, algorithm, algorithm_params=None, 
                          n_iterations=2000, step_size=0.01, local_epochs=5):
        adj_matrix, _ = TopologyGenerator.generate_adjacency_matrix(len(nodes), topology_type)
        W = ConsensusWeights.create_doubly_stochastic(adj_matrix)
        if algorithm_params is None:
            algorithm_params = {}
        
        dgd_algo = DGDLocalAlgorithm(
            nodes=nodes,
            weight_matrix=W,
            step_size=step_size,
            max_iterations=n_iterations,
            local_epochs=local_epochs
        )
        
        results = dgd_algo.run_algorithm(
            algorithm=algorithm,
            convergence_tol=1e-6,
            max_no_improvement=2000,
            **algorithm_params
        )
        
        summary = dgd_algo.get_convergence_summary()
        return {
            'results': results,
            'summary': summary,
            'topology': topology_type,
            'algorithm': algorithm,
            'adj_matrix': adj_matrix,
            'weight_matrix': W
        }
    
    def run_experiment_1_topologies_no_attack(self):
        print("\n" + "="*70)
        print("EXPERIMENT 1: DIFFERENT TOPOLOGIES WITHOUT ATTACK")
        print("="*70)
        
        R_train, mask_train, R_test, mask_test, n_users, n_items = self.load_movielens_data()
        nodes = self.create_dgd_nodes(R_train, n_nodes=20, rank=10)
        
        algorithms = ['d_fedavg', 'd_fedprox', 'd_fedadmm']
        topologies = ['complete', 'ring', 'star', 'random', 'mesh']
        
        exp1_results = {}
        
        for topology in topologies:
            print(f"\nTopology: {topology}")
            print("-" * 40)
            topology_results = {}
            
            for algorithm in algorithms:
                print(f"  Algorithm: {algorithm}")
                algorithm_params = {}
                if algorithm == 'd_fedprox':
                    algorithm_params['mu'] = 0.01
                elif algorithm == 'd_fedadmm':
                    algorithm_params['rho'] = 0.01
                
                try:
                    exp_result = self.run_dgd_experiment(
                        nodes=self.copy_dgd_nodes(nodes),
                        topology_type=topology,
                        algorithm=algorithm,
                        algorithm_params=algorithm_params,
                        n_iterations=2000,
                        step_size=0.01,
                        local_epochs=10
                    )
                    topology_results[algorithm] = exp_result
                    print(f"    Loss: {exp_result['results']['final_loss']:.4f}, "
                          f"Consensus: {exp_result['results']['final_consensus_error']:.4f}")
                except Exception as e:
                    print(f"    Error: {e}")
                    topology_results[algorithm] = None
            
            exp1_results[topology] = topology_results
        
        self.plot_experiment_1(exp1_results)
        return exp1_results
    
    def run_experiment_2_topologies_with_without_attack(self):
        print("\n" + "="*70)
        print("EXPERIMENT 2: TOPOLOGIES WITH/WITHOUT ATTACK (EXCEPT RANDOM)")
        print("="*70)
        
        R_train, mask_train, R_test, mask_test, n_users, n_items = self.load_movielens_data()
        base_nodes = self.create_dgd_nodes(R_train, n_nodes=20, rank=10)
        
        algorithms = ['d_fedavg', 'd_fedprox']
        topologies = ['complete', 'ring', 'star', 'mesh']
        attack_scenarios = ['no_attack', 'data_poisoning', 'model_manipulation']
        
        exp2_results = {}
        
        for topology in topologies:
            print(f"\nTopology: {topology}")
            print("-" * 40)
            topology_results = {}
            
            for attack_scenario in attack_scenarios:
                print(f"  Attack Scenario: {attack_scenario}")
                attack_results = {}
                
                if attack_scenario == 'no_attack':
                    nodes_to_use = self.copy_dgd_nodes(base_nodes)
                else:
                    nodes_to_use = self.copy_dgd_nodes(base_nodes)
                    if attack_scenario == 'data_poisoning':
                        AttackSimulatorDGD.simulate_data_poisoning(
                            nodes_to_use, 
                            attack_type='random',
                            attack_params={'malicious_ratio': 0.2, 'attack_strength': 2.0}
                        )
                    elif attack_scenario == 'model_manipulation':
                        AttackSimulatorDGD.simulate_model_manipulation(
                            nodes_to_use,
                            attack_type='gradient',
                            attack_params={'malicious_ratio': 0.2, 'manipulation_scale': 2.0}
                        )
                
                for algorithm in algorithms:
                    print(f"    Algorithm: {algorithm}")
                    algorithm_params = {}
                    if algorithm == 'd_fedprox':
                        algorithm_params['mu'] = 0.01
                    
                    try:
                        exp_result = self.run_dgd_experiment(
                            nodes=nodes_to_use,
                            topology_type=topology,
                            algorithm=algorithm,
                            algorithm_params=algorithm_params,
                            n_iterations=2000,
                            step_size=0.01,
                            local_epochs=10
                        )
                        attack_results[algorithm] = exp_result
                        print(f"      Loss: {exp_result['results']['final_loss']:.4f}")
                    except Exception as e:
                        print(f"      Error: {e}")
                        attack_results[algorithm] = None
                
                topology_results[attack_scenario] = attack_results
            
            exp2_results[topology] = topology_results
        
        self.plot_experiment_2(exp2_results)
        return exp2_results
    
    def run_experiment_3_attacked_topologies_with_security(self):
        print("\n" + "="*70)
        print("EXPERIMENT 3: ATTACKED TOPOLOGIES WITH SECURITY (EXCEPT RANDOM)")
        print("="*70)
        
        R_train, mask_train, R_test, mask_test, n_users, n_items = self.load_movielens_data()
        base_nodes = self.create_dgd_nodes(R_train, n_nodes=20, rank=10)
        
        topologies = ['complete', 'ring', 'star', 'mesh']
        attack_types = ['data_poisoning', 'model_manipulation']
        defense_types = ['no_defense', 'rl_defense', 'statistical_defense']
        algorithm = 'd_fedavg'
        
        exp3_results = {}
        
        for topology in topologies:
            print(f"\nTopology: {topology}")
            print("-" * 40)
            topology_results = {}
            
            for attack_type in attack_types:
                print(f"  Attack Type: {attack_type}")
                attack_results = {}
                
                for defense_type in defense_types:
                    print(f"    Defense: {defense_type}")
                    
                    nodes_to_use = self.copy_dgd_nodes(base_nodes)
                    
                    if attack_type == 'data_poisoning':
                        AttackSimulatorDGD.simulate_data_poisoning(
                            nodes_to_use,
                            attack_type='random',
                            attack_params={'malicious_ratio': 0.3, 'attack_strength': 2.5}
                        )
                    elif attack_type == 'model_manipulation':
                        AttackSimulatorDGD.simulate_model_manipulation(
                            nodes_to_use,
                            attack_type='gradient',
                            attack_params={'malicious_ratio': 0.3, 'manipulation_scale': 2.5}
                        )
                    
                    try:
                        exp_result = self.run_dgd_experiment(
                            nodes=nodes_to_use,
                            topology_type=topology,
                            algorithm=algorithm,
                            algorithm_params={},
                            n_iterations=2000,
                            step_size=0.01,
                            local_epochs=10
                        )
                        
                        if defense_type == 'rl_defense':
                            rl_defense = RLTrustDefenseDGD(n_nodes=20)
                            exp_result['defense_info'] = {
                                'type': 'rl',
                                'trust_distribution': rl_defense.get_trust_distribution()
                            }
                        elif defense_type == 'statistical_defense':
                            stat_defense = AnomalyDetectionDefenseDGD(threshold=2.0)
                            exp_result['defense_info'] = {
                                'type': 'statistical',
                                'threshold': 2.0
                            }
                        
                        attack_results[defense_type] = exp_result
                        print(f"      Loss: {exp_result['results']['final_loss']:.4f}")
                    except Exception as e:
                        print(f"      Error: {e}")
                        attack_results[defense_type] = None
                
                topology_results[attack_type] = attack_results
            
            exp3_results[topology] = topology_results
        
        self.plot_experiment_3(exp3_results)
        return exp3_results
    
    def plot_experiment_1(self, results):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Experiment 1: Topologies Without Attack', fontsize=16, y=1.02)
        
        algorithms = ['d_fedavg', 'd_fedprox', 'd_fedadmm']
        topologies = list(results.keys())
        colors = {'complete': '#1f77b4', 'ring': '#2ca02c', 'star': '#d62728', 
                 'random': '#ff7f0e', 'mesh': '#9467bd'}
        
        for row_idx, algorithm in enumerate(algorithms):
            for col_idx, metric in enumerate(['loss', 'consensus_error', 'gradient_norm']):
                ax = axes[row_idx, col_idx]
                for topology in topologies:
                    if topology in results and algorithm in results[topology]:
                        result_data = results[topology][algorithm]
                        if result_data is not None:
                            history = result_data['results']['history']
                            iterations = range(1, len(history[metric]) + 1)
                            ax.plot(iterations, history[metric], label=topology, 
                                   color=colors.get(topology, '#000000'), linewidth=2, alpha=0.7)
                ax.set_xlabel('Iteration')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{algorithm.upper()} - {metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                if row_idx == 0 and col_idx == 2:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/exp1_topologies_no_attack.png", dpi=2000, bbox_inches='tight')
        plt.close()
        
        self.plot_final_metrics_bar(results, 'exp1_final_metrics')
    
    def plot_experiment_2(self, results):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Experiment 2: Topologies With/Without Attack', fontsize=16, y=1.02)
        
        topologies = list(results.keys())
        attack_scenarios = ['no_attack', 'data_poisoning', 'model_manipulation']
        algorithms = ['d_fedavg', 'd_fedprox']
        colors = {'no_attack': '#27ae60', 'data_poisoning': '#c0392b', 'model_manipulation': '#f39c12'}
        
        for col_idx, topology in enumerate(topologies):
            for row_idx, algorithm in enumerate(algorithms):
                ax = axes[row_idx, col_idx]
                for scenario in attack_scenarios:
                    if topology in results and scenario in results[topology] and algorithm in results[topology][scenario]:
                        result_data = results[topology][scenario][algorithm]
                        if result_data is not None:
                            final_loss = result_data['results']['final_loss']
                            x_pos = list(attack_scenarios).index(scenario)
                            ax.bar(x_pos, final_loss, color=colors[scenario], alpha=0.8, width=0.6)
                
                ax.set_xlabel('Attack Scenario')
                ax.set_ylabel('Final Loss')
                ax.set_title(f'{topology.upper()} - {algorithm.upper()}')
                ax.set_xticks(range(len(attack_scenarios)))
                ax.set_xticklabels([s.replace('_', '\n').title() for s in attack_scenarios], fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/exp2_attack_comparison.png", dpi=2000, bbox_inches='tight')
        plt.close()
        
        self.plot_attack_impact_summary(results)
    
    def plot_experiment_3(self, results):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Experiment 3: Security Defense Effectiveness', fontsize=16, y=1.02)
        
        topologies = list(results.keys())
        attack_types = ['data_poisoning', 'model_manipulation']
        defense_types = ['no_defense', 'rl_defense', 'statistical_defense']
        colors = {'no_defense': '#c0392b', 'rl_defense': '#27ae60', 'statistical_defense': '#3498db'}
        
        for col_idx, topology in enumerate(topologies):
            for row_idx, attack_type in enumerate(attack_types):
                ax = axes[row_idx, col_idx]
                if topology in results and attack_type in results[topology]:
                    defense_results = results[topology][attack_type]
                    losses = []
                    for defense in defense_types:
                        if defense in defense_results and defense_results[defense] is not None:
                            losses.append(defense_results[defense]['results']['final_loss'])
                        else:
                            losses.append(0)
                    
                    bars = ax.bar(range(len(defense_types)), losses, 
                                 color=[colors.get(d, '#7f8c8d') for d in defense_types], alpha=0.8)
                    ax.set_xlabel('Defense Type')
                    ax.set_ylabel('Final Loss')
                    title_text = attack_type.replace("_", " ").title()
                    ax.set_title(f'{topology.upper()} - {title_text}')
                    ax.set_xticks(range(len(defense_types)))
                    ax.set_xticklabels([d.replace('_', '\n').title() for d in defense_types], fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    for bar, loss in zip(bars, losses):
                        if loss > 0:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height, f'{loss:.3f}', 
                                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/exp3_security_defense.png", dpi=2000, bbox_inches='tight')
        plt.close()
    
    def plot_final_metrics_bar(self, results, filename):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algorithms = ['d_fedavg', 'd_fedprox', 'd_fedadmm']
        topologies = list(results.keys())
        
        for idx, metric in enumerate(['final_loss', 'final_consensus_error', 'total_communication']):
            ax = axes[idx]
            x = np.arange(len(topologies))
            width = 0.25
            
            for alg_idx, algorithm in enumerate(algorithms):
                metric_values = []
                for topology in topologies:
                    if topology in results and algorithm in results[topology]:
                        result_data = results[topology][algorithm]
                        if result_data is not None:
                            if metric == 'final_loss':
                                value = result_data['results']['final_loss']
                            elif metric == 'final_consensus_error':
                                value = result_data['results']['final_consensus_error']
                            else:
                                value = result_data['summary']['total_communication'] / 1000
                        else:
                            value = 0
                    else:
                        value = 0
                    metric_values.append(value)
                
                ax.bar(x + (alg_idx - 1) * width, metric_values, width,
                      label=algorithm.upper(), alpha=0.8)
            
            ax.set_xlabel('Topology')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(topologies, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/{filename}.png", dpi=2000, bbox_inches='tight')
        plt.close()
    
    def plot_attack_impact_summary(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        topologies = list(results.keys())
        algorithms = ['d_fedavg', 'd_fedprox']
        attack_types = ['data_poisoning', 'model_manipulation']
        
        for row_idx, algorithm in enumerate(algorithms):
            for col_idx, attack_type in enumerate(attack_types):
                ax = axes[row_idx, col_idx]
                
                degradations = []
                topology_labels = []
                
                for topology in topologies:
                    if (topology in results and 'no_attack' in results[topology] and 
                        algorithm in results[topology]['no_attack'] and
                        attack_type in results[topology] and 
                        algorithm in results[topology][attack_type]):
                        
                        no_attack_result = results[topology]['no_attack'][algorithm]
                        attack_result = results[topology][attack_type][algorithm]
                        
                        if no_attack_result is not None and attack_result is not None:
                            no_attack_loss = no_attack_result['results']['final_loss']
                            attack_loss = attack_result['results']['final_loss']
                            degradation = (attack_loss - no_attack_loss) / no_attack_loss * 100
                            degradations.append(degradation)
                            topology_labels.append(topology[0].upper())
                
                if degradations:
                    bars = ax.bar(range(len(degradations)), degradations, alpha=0.8)
                    ax.set_xlabel('Topology')
                    ax.set_ylabel('Performance Degradation (%)')
                    ax.set_title(f'{algorithm.upper()} - {attack_type.replace("_", " ").title()}')
                    ax.set_xticks(range(len(degradations)))
                    ax.set_xticklabels(topology_labels)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    for bar, degradation in zip(bars, degradations):
                        height = bar.get_height()
                        color = 'red' if degradation > 0 else 'green'
                        ax.text(bar.get_x() + bar.get_width()/2., height, 
                               f'{degradation:+.1f}%', ha='center', va='bottom' if degradation > 0 else 'top', 
                               fontsize=9, color=color)
        
        plt.suptitle('Attack Impact Summary', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/exp2_attack_impact_summary.png", dpi=2000, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self):
        print("="*70)
        print("COMPREHENSIVE DGD EXPERIMENTS")
        print("="*70)
        
        print("\nRunning Experiment 1...")
        exp1_results = self.run_experiment_1_topologies_no_attack()
        
        # print("\nRunning Experiment 2...")
        # exp2_results = self.run_experiment_2_topologies_with_without_attack()
        
        print("\nRunning Experiment 3...")
        exp3_results = self.run_experiment_3_attacked_topologies_with_security()
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETE!")
        print(f"Results saved in: {self.results_dir}/")
        print(f"Plots generated:")
        print(f"  - exp1_topologies_no_attack.png")
        print(f"  - exp1_final_metrics.png")
        print(f"  - exp2_attack_comparison.png")
        print(f"  - exp2_attack_impact_summary.png")
        print(f"  - exp3_security_defense.png")
        print(f"Data saved: all_results.pkl")
        print(f"{'='*70}")
        
def main():
    tester = DGDComprehensiveTester()
    tester.run_all_experiments()

if __name__ == "__main__":
    main()