import numpy as np
import matplotlib.pyplot as plt
from models.gossip_learning import GossipNode, GossipNetwork, EmbeddingClipping
from data.movielens_loader import MovieLensLoader
from data.data_preprocessor import DataPreprocessor
from security.attack import AttackSimulator
from security.defense import RLTrustDefense, AnomalyDetectionDefense
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pickle

os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())

def create_results_directory():
    os.makedirs('results', exist_ok=True)

def prepare_movielens_data():
    cache_file = 'results/movielens_cache.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    loader = MovieLensLoader("dataset")
    data = loader.load_dataset("MovieLens100K")
    R, mask = loader.get_ratings_matrix(data['ratings_df'])
    preprocessor = DataPreprocessor(R, mask)
    R_train, mask_train, R_test, mask_test = preprocessor.split_train_test_by_user(n_test_per_user=10)
    
    result = (R_train, mask_train, R_test, mask_test)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    return result

def create_base_nodes(R_train, n_nodes=20, rank=10):
    n_users, n_items = R_train.shape
    user_groups = np.array_split(np.arange(n_users), n_nodes)
    base_nodes_data = []
    
    for i, user_indices in enumerate(user_groups):
        if len(user_indices) > 0:
            user_idx = user_indices[0]
            node_ratings = R_train[user_idx:user_idx+1, :].copy()
        else:
            node_ratings = np.full((1, n_items), np.nan)
        
        node_data = {
            'node_id': i,
            'user_ratings': node_ratings,
            'n_items': n_items,
            'rank': rank,
            'R_min': 0.5,
            'R_max': 5.0,
            'use_clipping': True
        }
        base_nodes_data.append(node_data)
    
    return base_nodes_data

def create_node_from_data(node_data):
    return GossipNode(
        node_id=node_data['node_id'],
        user_ratings=node_data['user_ratings'].copy(),
        n_items=node_data['n_items'],
        rank=node_data['rank'],
        R_min=node_data['R_min'],
        R_max=node_data['R_max'],
        use_clipping=node_data['use_clipping']
    )

def compute_test_rmse(nodes, R_test):
    test_loss = 0.0
    test_count = 0
    for node in nodes:
        user_idx = node.node_id
        if user_idx < R_test.shape[0]:
            user_test_ratings = np.full((1, R_test.shape[1]), np.nan)
            user_test_ratings[0, :] = R_test[user_idx, :]
            node_test_loss = node.get_validation_loss(user_test_ratings)
            n_test_ratings = np.sum(~np.isnan(user_test_ratings))
            if n_test_ratings > 0:
                test_loss += float(node_test_loss) * n_test_ratings
                test_count += n_test_ratings
    if test_count > 0:
        avg_test_loss = test_loss / test_count
        return np.sqrt(avg_test_loss)
    return np.nan

def run_single_gossip_experiment(args):
    topology, base_nodes_data, R_test, n_cycles, compression_ratio, attack_type, attack_params, defense_type = args
    
    nodes = [create_node_from_data(data) for data in base_nodes_data]
    
    network = GossipNetwork(
        nodes=nodes,
        topology=topology,
        cycle_time=100.0,
        online_probability=0.9,
        use_clipping=True
    )
    
    if attack_type == 'data_poisoning':
        for node in nodes:
            poisoned_ratings = AttackSimulator.simulate_data_poisoning(
                node.user_ratings,
                attack_type='random',
                attack_params=attack_params
            )
            node.user_ratings = poisoned_ratings
    
    results = network.run_experiment(
        n_cycles=n_cycles,
        compression_ratio=compression_ratio,
        local_epochs=1,
        merge_method='age_weighted',
    )
    
    if attack_type == 'model_manipulation':
        attack_count = 0
        for node in nodes:
            if np.random.random() < 0.3: 
                manipulated_u = AttackSimulator.simulate_model_attack(
                    node.u,
                    attack_type='gradient',
                    attack_params={'manipulation_scale': 2.0}
                )
                node.u = manipulated_u
                attack_count += 1
    
    filtered_updates = None
    if defense_type and (attack_type is not None):
        if defense_type == 'rl':
            defense_system = RLTrustDefense(n_nodes=len(nodes), learning_rate=0.1, gamma=0.95, epsilon=0.1)
        elif defense_type == 'stat':
            defense_system = AnomalyDetectionDefense(threshold=2.0, window_size=10)
        else:
            defense_system = None
            
        if defense_system:
            updates = {}
            for i, node in enumerate(nodes):
                if np.random.random() < 0.8:  
                    updates[i] = node.u.copy()
            
            filtered_updates = defense_system.defend(updates, global_consensus=None)
            
            for node_id, filtered_update in filtered_updates.items():
                if node_id < len(nodes):
                    nodes[node_id].u = filtered_update
    
    test_rmse = compute_test_rmse(nodes, R_test)
    final_loss = float(results['final_metrics']['loss'])
    final_consensus = float(results['final_metrics']['consensus_error'])
    total_comm = sum(results['history']['communication_cost'])
    
    return {
        'topology': topology,
        'attack_type': attack_type,
        'defense_type': defense_type,
        'final_loss': final_loss,
        'test_rmse': test_rmse,
        'final_consensus': final_consensus,
        'total_communication': total_comm,
        'loss_history': [float(l) for l in results['history']['loss'][:n_cycles]],
        'consensus_history': [float(c) for c in results['history']['consensus_error'][:n_cycles]],
        'comm_history': results['history']['communication_cost'][:n_cycles],
        'filtered_updates': filtered_updates
    }

def run_parallel_experiments(experiment_args_list, max_workers=None):
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(run_single_gossip_experiment, args): args for args in experiment_args_list}
        
        completed = 0
        total = len(experiment_args_list)
        
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            try:
                result = future.result()
                topology = result['topology']
                attack_type = result['attack_type'] or 'no_attack'
                defense_type = result['defense_type']
                
                if topology not in results:
                    results[topology] = {}
                
                if defense_type:
                    key = f"{attack_type}_{defense_type}_defense"
                else:
                    key = attack_type
                
                results[topology][key] = result
                
                completed += 1
                print(f"Completed {completed}/{total}")
                    
            except Exception as e:
                print(f"Experiment failed: {e}")
                continue
    
    return results

def test_gossip_no_attack_parallel(R_train, R_test, base_nodes_data, n_cycles=40):
    print("="*70)
    print(f"GOSSIP WITHOUT ATTACK")
    print("="*70)
    
    topologies = ['complete', 'ring', 'star', 'random']
    experiment_args = []
    
    for topology in topologies:
        experiment_args.append((topology, base_nodes_data, R_test, n_cycles, 0.2, None, None, None))
    
    print(f"Running {len(experiment_args)} experiments")
    start_time = time.time()
    results = run_parallel_experiments(experiment_args)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.2f} seconds")
    for topology in topologies:
        if topology in results and 'no_attack' in results[topology]:
            res = results[topology]['no_attack']
            print(f"{topology}: Loss: {res['final_loss']:.4f}, RMSE: {res['test_rmse']:.4f}")
    
    return results

def test_gossip_with_attacks_parallel(R_train, R_test, base_nodes_data, n_cycles=40):
    print("\n" + "="*70)
    print("GOSSIP WITH ATTACKS")
    print("="*70)
    
    topologies = ['complete', 'star', 'ring']
    attack_types = ['data_poisoning', 'model_manipulation', None]
    attack_params = {'malicious_ratio': 0.2, 'attack_strength': 4.0, 'manipulation_scale': 2.0}
    
    experiment_args = []
    
    for topology in topologies:
        for attack_type in attack_types:
            experiment_args.append((topology, base_nodes_data, R_test, n_cycles, 0.2, attack_type, attack_params, None))
    
    print(f"Running {len(experiment_args)} experiments")
    start_time = time.time()
    results = run_parallel_experiments(experiment_args)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.2f} seconds")
    for topology in topologies:
        if topology in results:
            print(f"{topology}:")
            for attack_key in ['no_attack', 'data_poisoning', 'model_manipulation']:
                if attack_key in results[topology]:
                    res = results[topology][attack_key]
                    attack_label = 'No Attack' if attack_key == 'no_attack' else attack_key
                    print(f"  {attack_label}: Loss: {res['final_loss']:.4f}")
    
    return results

def test_gossip_with_defense_parallel(R_train, R_test, base_nodes_data, n_cycles=40):
    print("\n" + "="*70)
    print("GOSSIP WITH DEFENSE")
    print("="*70)
    
    topologies = ['complete', 'star', 'ring']
    attack_types = ['data_poisoning', 'model_manipulation']
    attack_params = {'malicious_ratio': 0.2, 'attack_strength': 4.0, 'manipulation_scale': 2.0}
    defense_types = [None, 'rl', 'stat']
    
    experiment_args = []
    
    for topology in topologies:
        for attack_type in attack_types:
            for defense_type in defense_types:
                experiment_args.append((topology, base_nodes_data, R_test, n_cycles, 0.2, attack_type, attack_params, defense_type))
    
    print(f"Running {len(experiment_args)} experiments")
    start_time = time.time()
    results = run_parallel_experiments(experiment_args)
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.2f} seconds")
    for topology in topologies:
        if topology in results:
            print(f"{topology}:")
            for attack_type in attack_types:
                print(f"  {attack_type}:")
                
                key_no_defense = attack_type
                if key_no_defense in results[topology]:
                    res = results[topology][key_no_defense]
                    print(f"    No defense: Loss: {res['final_loss']:.4f}")
                
                key_rl = f"{attack_type}_rl_defense"
                if key_rl in results[topology]:
                    res = results[topology][key_rl]
                    print(f"    RL defense: Loss: {res['final_loss']:.4f}")
                
                key_stat = f"{attack_type}_stat_defense"
                if key_stat in results[topology]:
                    res = results[topology][key_stat]
                    print(f"    Stat defense: Loss: {res['final_loss']:.4f}")
    
    return results

def save_plot_no_attack(results_no_attack, n_cycles=40):
    topologies = list(results_no_attack.keys())
    colors = {'complete': '#1f77b4', 'ring': '#2ca02c', 'star': '#d62728', 'random': '#ff7f0e'}
    markers = {'complete': 'o', 'ring': 's', 'star': '^', 'random': 'D'}
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for topology in topologies:
        if 'no_attack' in results_no_attack[topology]:
            loss_history = results_no_attack[topology]['no_attack']['loss_history']
            cycles = range(min(len(loss_history), n_cycles))
            plt.plot(cycles, loss_history[:n_cycles], label=topology, color=colors[topology], marker=markers[topology], markersize=4, markevery=max(1, n_cycles//10), linewidth=2, alpha=0.8)
    plt.xlabel('Gossip Cycle')
    plt.ylabel('Loss')
    plt.title(f'Loss Convergence (No Attack)')
    plt.legend()
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    for topology in topologies:
        if 'no_attack' in results_no_attack[topology]:
            consensus_history = results_no_attack[topology]['no_attack']['consensus_history']
            cycles = range(min(len(consensus_history), n_cycles))
            plt.plot(cycles, consensus_history[:n_cycles], label=topology, color=colors[topology], marker=markers[topology], markersize=4, markevery=max(1, n_cycles//10), linewidth=2, alpha=0.8)
    plt.xlabel('Gossip Cycle')
    plt.ylabel('Consensus Error')
    plt.title(f'Consensus Error (No Attack)')
    plt.legend()
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    final_losses = []
    test_rmses = []
    valid_topologies = []
    
    for topology in topologies:
        if 'no_attack' in results_no_attack[topology]:
            final_losses.append(results_no_attack[topology]['no_attack']['final_loss'])
            test_rmses.append(results_no_attack[topology]['no_attack']['test_rmse'])
            valid_topologies.append(topology)
    
    x = np.arange(len(valid_topologies))
    width = 0.35
    bars1 = plt.bar(x - width/2, final_losses, width, label='Final Loss', color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = plt.bar(x + width/2, test_rmses, width, label='Test RMSE', color='#e74c3c', edgecolor='black', alpha=0.8)
    plt.xlabel('Topology')
    plt.ylabel('Error Value')
    plt.title('Final Performance Metrics')
    plt.xticks(x, valid_topologies)
    plt.legend()
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 2, 4)
    comm_costs = []
    convergence_speed = []
    valid_topologies = []
    
    for topology in topologies:
        if 'no_attack' in results_no_attack[topology]:
            comm_costs.append(results_no_attack[topology]['no_attack']['total_communication'])
            loss_history = results_no_attack[topology]['no_attack']['loss_history']
            threshold = 0.5 * loss_history[0] if loss_history else 0
            converged_cycles = None
            for i, loss in enumerate(loss_history):
                if loss <= threshold:
                    converged_cycles = i + 1
                    break
            convergence_speed.append(converged_cycles if converged_cycles else len(loss_history))
            valid_topologies.append(topology)
    
    x = np.arange(len(valid_topologies))
    ax1 = plt.gca()
    bars1 = ax1.bar(x - 0.2, comm_costs, width=0.4, label='Comm Cost', color='#3498db', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Topology')
    ax1.set_ylabel('Communication Cost (bytes)', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, convergence_speed, width=0.4, label='Conv Speed', color='#2ca02c', edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Cycles to 50% Loss', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    plt.title('Communication vs Convergence')
    plt.xticks(x, valid_topologies)
    
    plt.suptitle(f'Topology Comparison (No Attack) - {n_cycles} Cycles', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/plot1_topology_comparison_no_attack.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def save_plot_attack_impact(results_with_attack, n_cycles=40):
    fig = plt.figure(figsize=(15, 10))
    topologies = list(results_with_attack.keys())
    attack_types = ['no_attack', 'data_poisoning', 'model_manipulation']
    colors = {'no_attack': '#27ae60', 'data_poisoning': '#c0392b', 'model_manipulation': '#f39c12'}
    linestyles = {'no_attack': '-', 'data_poisoning': '--', 'model_manipulation': ':'}
    markers = {'no_attack': 'o', 'data_poisoning': 's', 'model_manipulation': '^'}
    
    for idx, topology in enumerate(topologies, 1):
        plt.subplot(2, 3, idx)
        
        for attack_type in attack_types:
            if attack_type in results_with_attack[topology]:
                loss_history = results_with_attack[topology][attack_type]['loss_history']
                cycles = range(min(len(loss_history), n_cycles))
                plt.plot(cycles, loss_history[:n_cycles], label=attack_type.replace('_', ' ').title(),
                        color=colors[attack_type], linestyle=linestyles[attack_type],
                        marker=markers[attack_type], markersize=4, markevery=5,
                        linewidth=2, alpha=0.8)
        
        plt.xlabel('Gossip Cycle')
        plt.ylabel('Loss')
        plt.title(f'{topology.upper()} Topology')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.yscale('log')
    
    plt.subplot(2, 3, 4)
    final_performance = {}
    bar_colors = []
    
    for topology in topologies:
        for attack_type in attack_types:
            if attack_type in results_with_attack[topology]:
                key = f"{topology}\n{attack_type.replace('_', ' ').title()}"
                final_performance[key] = results_with_attack[topology][attack_type]['final_loss']
                bar_colors.append(colors[attack_type])
    
    keys = list(final_performance.keys())
    values = list(final_performance.values())
    x_pos = np.arange(len(keys))
    
    bars = plt.bar(x_pos, values, color=bar_colors, edgecolor='black', alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x_pos, keys, rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.subplot(2, 3, 5)
    degradation_data = []
    labels = []
    
    for topology in topologies:
        if 'no_attack' in results_with_attack[topology]:
            base_loss = results_with_attack[topology]['no_attack']['final_loss']
            
            for attack_type in ['data_poisoning', 'model_manipulation']:
                if attack_type in results_with_attack[topology]:
                    attack_loss = results_with_attack[topology][attack_type]['final_loss']
                    degradation = (attack_loss - base_loss) / base_loss * 100
                    degradation_data.append(degradation)
                    labels.append(f"{topology[0].upper()}\n{attack_type[0].upper()}")
    
    colors_degradation = ['#e74c3c', '#f39c12'] * len(topologies)
    bars = plt.bar(range(len(degradation_data)), degradation_data, color=colors_degradation, edgecolor='black', alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Performance Degradation (%)')
    plt.title('Attack Impact Comparison')
    plt.xticks(range(len(degradation_data)), labels, fontsize=9)
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    for bar, degradation in zip(bars, degradation_data):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{degradation:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 3, 6)
    rmse_data = []
    rmse_labels = []
    rmse_colors = []
    
    for topology in topologies:
        for attack_type in attack_types:
            if attack_type in results_with_attack[topology]:
                rmse = results_with_attack[topology][attack_type]['test_rmse']
                rmse_data.append(rmse)
                rmse_labels.append(f"{topology[0].upper()}\n{attack_type[0].upper()}")
                rmse_colors.append(colors[attack_type])
    
    bars = plt.bar(range(len(rmse_data)), rmse_data, color=rmse_colors, edgecolor='black', alpha=0.8)
    plt.xlabel('Configuration')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE Comparison')
    plt.xticks(range(len(rmse_data)), rmse_labels, fontsize=9)
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    for bar, rmse in zip(bars, rmse_data):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{rmse:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.suptitle(f'Attack Impact Analysis - {n_cycles} Cycles', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/plot2_attack_impact_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def save_plot_defense_efficacy(results_with_defense, n_cycles=40):
    fig = plt.figure(figsize=(15, 12))
    topologies = list(results_with_defense.keys())
    attack_types = ['data_poisoning', 'model_manipulation']
    defense_types = ['no_defense', 'rl_defense', 'stat_defense']
    defense_colors = {'no_defense': '#c0392b', 'rl_defense': '#27ae60', 'stat_defense': '#3498db'}
    
    for attack_idx, attack_type in enumerate(attack_types, 1):
        plt.subplot(3, 3, attack_idx)
        
        for topology in topologies:
            topology_data = []
            defense_labels = []
            
            for defense_type in defense_types:
                config_key = f'{attack_type}_{defense_type}'
                if config_key in results_with_defense[topology]:
                    loss = results_with_defense[topology][config_key]['final_loss']
                    topology_data.append(loss)
                    defense_labels.append(defense_type.replace('_', ' ').title())
            
            x_pos = np.arange(len(topology_data))
            plt.bar(x_pos, topology_data, width=0.25, label=topology, color=[defense_colors[d.lower().replace(' ', '_')] for d in defense_labels], alpha=0.8, edgecolor='black')
        
        plt.xlabel('Defense Type')
        plt.ylabel('Final Loss')
        plt.title(f'{attack_type.replace("_", " ").title()} Attack')
        plt.xticks(x_pos, defense_labels, rotation=45, ha='right', fontsize=9)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    for attack_idx, attack_type in enumerate(attack_types, 4):
        plt.subplot(3, 3, attack_idx)
        
        improvement_data = []
        improvement_labels = []
        
        for topology in topologies:
            config_no_defense = f'{attack_type}_no_defense'
            config_rl_defense = f'{attack_type}_rl_defense'
            config_stat_defense = f'{attack_type}_stat_defense'
            
            if config_no_defense in results_with_defense[topology] and config_rl_defense in results_with_defense[topology]:
                loss_no_defense = results_with_defense[topology][config_no_defense]['final_loss']
                loss_rl_defense = results_with_defense[topology][config_rl_defense]['final_loss']
                improvement_rl = (loss_no_defense - loss_rl_defense) / loss_no_defense * 100
                improvement_data.append(improvement_rl)
                improvement_labels.append(f"{topology[0].upper()}\nRL")
            
            if config_no_defense in results_with_defense[topology] and config_stat_defense in results_with_defense[topology]:
                loss_no_defense = results_with_defense[topology][config_no_defense]['final_loss']
                loss_stat_defense = results_with_defense[topology][config_stat_defense]['final_loss']
                improvement_stat = (loss_no_defense - loss_stat_defense) / loss_no_defense * 100
                improvement_data.append(improvement_stat)
                improvement_labels.append(f"{topology[0].upper()}\nStat")
        
        colors_improvement = ['#27ae60', '#3498db'] * len(topologies)
        bars = plt.bar(range(len(improvement_data)), improvement_data, color=colors_improvement, edgecolor='black', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('Improvement (%)')
        plt.title(f'{attack_type.replace("_", " ").title()} Defense Improvement')
        plt.xticks(range(len(improvement_data)), improvement_labels, fontsize=9)
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
        
        for bar, improvement in zip(bars, improvement_data):
            height = bar.get_height()
            color = 'green' if improvement > 0 else 'red'
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{improvement:+.1f}%', ha='center', va='bottom' if improvement > 0 else 'top', fontsize=9, color=color)
    
    for attack_idx, attack_type in enumerate(attack_types, 7):
        plt.subplot(3, 3, attack_idx)
        
        rmse_data = []
        rmse_labels = []
        bar_colors = []
        
        for topology in topologies:
            for defense_type in defense_types:
                config_key = f'{attack_type}_{defense_type}'
                if config_key in results_with_defense[topology]:
                    rmse = results_with_defense[topology][config_key]['test_rmse']
                    rmse_data.append(rmse)
                    rmse_labels.append(f"{topology[0].upper()}\n{defense_type[0].upper()}")
                    if defense_type == 'no_defense':
                        bar_colors.append(defense_colors['no_defense'])
                    elif defense_type == 'rl_defense':
                        bar_colors.append(defense_colors['rl_defense'])
                    elif defense_type == 'stat_defense':
                        bar_colors.append(defense_colors['stat_defense'])
        
        bars = plt.bar(range(len(rmse_data)), rmse_data, color=bar_colors, edgecolor='black', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('Test RMSE')
        plt.title(f'{attack_type.replace("_", " ").title()} Test RMSE')
        plt.xticks(range(len(rmse_data)), rmse_labels, fontsize=9)
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
        
        for bar, rmse in zip(bars, rmse_data):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{rmse:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.suptitle(f'Defense System Efficacy - {n_cycles} Cycles', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/plot3_defense_efficacy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def run_all_experiments_fast():
    create_results_directory()
    
    print("="*70)
    print("GOSSIP LEARNING SECURITY ANALYSIS")
    print("="*70)
    
    start_time = time.time()
    
    print("Loading MovieLens data...")
    R_train, mask_train, R_test, mask_test = prepare_movielens_data()
    
    n_nodes = 20
    rank = 10
    n_cycles = 40
    
    print(f"Configuration: {n_nodes} nodes, rank={rank}, {n_cycles} cycles")
    print(f"Using {min(mp.cpu_count(), 8)} CPU cores")
    
    print("Creating base nodes...")
    base_nodes_data = create_base_nodes(R_train, n_nodes=n_nodes, rank=rank)
    
    # print("="*70)
    # print("EXPERIMENT 1: TOPOLOGY COMPARISON")
    # print("="*70)
    # results_no_attack = test_gossip_no_attack_parallel(R_train, R_test, base_nodes_data, n_cycles=n_cycles)
    
    # print("\n" + "="*70)
    # print("EXPERIMENT 2: ATTACK IMPACT")
    # print("="*70)
    # results_with_attack = test_gossip_with_attacks_parallel(R_train, R_test, base_nodes_data, n_cycles=n_cycles)
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: DEFENSE SYSTEM")
    print("="*70)
    results_with_defense = test_gossip_with_defense_parallel(R_train, R_test, base_nodes_data, n_cycles=n_cycles)
    
    print("\n" + "="*70)
    print("SAVING PLOTS...")
    print("="*70)
    
    # save_plot_no_attack(results_no_attack, n_cycles=n_cycles)
    # print("Plot 1 saved: topology_comparison_no_attack.png")
    
    # save_plot_attack_impact(results_with_attack, n_cycles=n_cycles)
    # print("Plot 2 saved: attack_impact_comparison.png")
    
    save_plot_defense_efficacy(results_with_defense, n_cycles=n_cycles)
    print("Plot 3 saved: defense_efficacy_comparison.png")
    
    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # with open('results/all_results.pkl', 'wb') as f:
    #     pickle.dump({'no_attack': results_no_attack, 'with_attack': results_with_attack, 'with_defense': results_with_defense, 'config': {'n_nodes': n_nodes, 'rank': rank, 'n_cycles': n_cycles}}, f)
    
def main():
    print("Initializing analysis...")
    print(f"CPU cores: {mp.cpu_count()}")
    
    U_test = np.random.randn(10, 5) * 3
    V_test = np.random.randn(100, 5) * 3
    U_clipped, V_clipped = EmbeddingClipping.clip_embeddings(U_test, V_test, R_max=5.0)
    
    print("="*70)
    print("STARTING ANALYSIS")
    print("="*70)
    
    run_all_experiments_fast()
    
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()