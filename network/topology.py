import numpy as np
import networkx as nx

class TopologyGenerator:
    @staticmethod
    def generate_adjacency_matrix(n_nodes, topology='ring', seed=None):
        if seed is not None:
            np.random.seed(seed)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        if topology == 'complete':
            adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        elif topology == 'ring':
            for i in range(n_nodes):
                adj_matrix[i, (i-1) % n_nodes] = 1
                adj_matrix[i, (i+1) % n_nodes] = 1
        elif topology == 'star':
            adj_matrix[0, 1:] = 1
            adj_matrix[1:, 0] = 1
        elif topology == 'random':
            p = 2 * np.log(n_nodes) / n_nodes
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if np.random.random() < p:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
        elif topology == 'mesh':
            k = min(4, n_nodes-1)
            for i in range(n_nodes):
                neighbours = np.random.choice([j for j in range(n_nodes) if j != i], 
                                             size=min(k, n_nodes-1), replace=False)
                for j in neighbours:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        else:
            raise ValueError(f"Topology {topology} not supported")
        graph = nx.from_numpy_array(adj_matrix)
        return adj_matrix.astype(int), graph
    
    @staticmethod
    def visualize_topology(adj_matrix, save_path):
        import matplotlib.pyplot as plt
        graph = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(graph)
        plt.figure(figsize=(8, 8))
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
                node_size=500, font_size=10, font_weight='bold')
        plt.title(f"Network Topology ({len(graph.nodes)} nodes)")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def check_connectivity(adj_matrix):
        graph = nx.from_numpy_array(adj_matrix)
        return nx.is_connected(graph)
    
    @staticmethod
    def generate_time_varying_topology(n_nodes, base_topology='ring', variations=3, seed=None):
        if seed is not None:
            np.random.seed(seed)
        topologies = []
        for _ in range(variations):
            if base_topology == 'ring':
                adj_matrix = np.zeros((n_nodes, n_nodes))
                for i in range(n_nodes):
                    left = (i - 1 - np.random.randint(0, 2)) % n_nodes
                    right = (i + 1 + np.random.randint(0, 2)) % n_nodes
                    adj_matrix[i, left] = 1
                    adj_matrix[i, right] = 1
                    adj_matrix[left, i] = 1
                    adj_matrix[right, i] = 1
            elif base_topology == 'random_varying':
                p = 2 * np.log(n_nodes) / n_nodes * np.random.uniform(0.8, 1.2)
                adj_matrix = np.zeros((n_nodes, n_nodes))
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if np.random.random() < p:
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 1
            else:
                adj_matrix, _ = TopologyGenerator.generate_adjacency_matrix(n_nodes, base_topology)
            topologies.append(adj_matrix)
        return topologies
    
    @staticmethod
    def generate_periodic_strongly_connected(n_nodes, period=5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        topologies = []
        for t in range(period):
            adj_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                forward = (i + t) % n_nodes
                backward = (i - t) % n_nodes
                adj_matrix[i, forward] = 1
                adj_matrix[i, backward] = 1
                adj_matrix[forward, i] = 1
                adj_matrix[backward, i] = 1
            topologies.append(adj_matrix)
        return topologies