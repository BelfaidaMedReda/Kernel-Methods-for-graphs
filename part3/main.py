import networkx as nx
import numpy as np
from itertools import combinations
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_dataset():
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')

    graphs = [
        to_networkx(
            data,
            to_undirected=True,        
            node_attrs=['x'],          
            edge_attrs=['edge_attr']   
        )
        for data in dataset
    ]

    y = dataset.y.tolist()
    return graphs, y


def get_graphlets(g):
    g3,g4 = nx.Graph(), nx.Graph()
    g4.add_nodes_from([1,2,3])
    g3.add_nodes_from([1,2,3])
    g3.add_edge(2,3)
    three_size_graphlets = [nx.complete_graph(3), nx.path_graph(3), g3, g4]
    res = [0]*4
    for comb in combinations(g.nodes(),3):
        subgraph = g.subgraph(comb)
        for i in range(4) :
            if nx.is_isomorphic(subgraph,three_size_graphlets[i]):
                res[i] +=1
                break
    return res


def graphlet_kernel(G_train, G_test):
    n_train = len(G_train)
    n_test = len(G_test)
    train_graphlets_count = np.array([get_graphlets(graph) for graph in G_train])
    test_graphlets_count = np.array([get_graphlets(graph) for graph in G_test])
    K_train = [ [ np.dot(train_graphlets_count[i], train_graphlets_count[j]) for j in range(n_train) ] for i in range(n_train) ]
    K_test = [ [ np.dot(test_graphlets_count[i], train_graphlets_count[j]) for j in range(n_train) ] for i in range(n_test) ]
    return np.array(K_train), np.array(K_test)


def shortest_path_counts(G, l_max):
    path_length_count = np.zeros(l_max, dtype=int)
    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=l_max))
    for source in lengths:
        for target in lengths[source]:
            length = lengths[source][target]
            if length <= l_max and source != target:
                path_length_count[length - 1] += 1
    path_length_count //= 2
    return path_length_count


def shortest_path_kernel(G_train, G_test, l_max=5):
    train_path_counts = np.array([shortest_path_counts(graph, l_max) for graph in G_train])
    test_path_counts = np.array([shortest_path_counts(graph, l_max) for graph in G_test])
    K_train = np.dot(train_path_counts, train_path_counts.T)
    K_test = np.dot(test_path_counts, train_path_counts.T)
    return K_train, K_test


def main():
    graphs, y = load_dataset()
    print(f"Loaded {len(graphs)} graphs from the MUTAG dataset.")
    for i, G in enumerate(graphs):
        print(f"Graph {i+1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, label: {y[i]}")

    # Split the dataset into training and testing sets
    G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(G_train)} graphs")
    print(f"Testing set: {len(G_test)} graphs")

    # Applying Graphlet_kernel
    K_train, K_test = graphlet_kernel(G_train, G_test)
    # print("Graphlet Kernel on training set:\n", K_train)
    # print("Graphlet Kernel on testing set:\n", K_test)
    # print("Matrix Kernel shape of train data :", K_train.shape)
    # print("Matrix Kernel shape of test data :", K_test.shape)

    # Evaluating Graphlet Kernel performance
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Graphlet Kernel SVM Accuracy: {accuracy * 100:.2f}%")

    # Applying Shortest Path Kernel
    K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)
    # print("Shortest Path Kernel on training set:\n", K_train_sp)
    # print("Shortest Path Kernel on testing set:\n", K_test_sp)
    # print("Matrix Kernel shape of train data :", K_train_sp.shape)
    # print("Matrix Kernel shape of test data :", K_test_sp.shape)

    # Evaluating Shortest Path Kernel performance
    clf_sp = SVC(kernel='precomputed')
    clf_sp.fit(K_train_sp, y_train)

    y_pred_sp = clf_sp.predict(K_test_sp)
    accuracy_sp = accuracy_score(y_test, y_pred_sp)
    print(f"Shortest Path Kernel SVM Accuracy: {accuracy_sp * 100:.2f}%")






if __name__ == "__main__":
    main()