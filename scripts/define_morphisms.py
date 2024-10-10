import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def define_morphisms(all_weights):
    G = nx.DiGraph()
    epochs = sorted(all_weights.keys())
    for i in range(len(epochs) - 1):
        source = epochs[i]
        target = epochs[i + 1]
        # Compute Euclidean distance as a placeholder for morphism "cost"
        distance = np.linalg.norm(all_weights[source] - all_weights[target])
        G.add_edge(source, target, weight=distance)
    return G

def main():
    all_weights = np.load('extracted_weights.npy', allow_pickle=True).item()
    G = define_morphisms(all_weights)
    nx.write_gpickle(G, 'model_graph.gpickle')
    print("Graph with morphisms defined and saved as model_graph.gpickle")

if __name__ == "__main__":
    main()