import networkx as nx
import numpy as np
import random
import scipy.sparse.linalg as spla
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv, eigs
from sklearn.cluster import KMeans


def spectral_clustering(G, k):
    W = nx.adjacency_matrix(G).astype(float)
    
    degrees = np.array(W.sum(axis=1)).flatten()
    D = diags(degrees)
    L = eye(len(degrees)) - inv(D) @ W

    #Calculate the first k eigenvectors of the Laplacian matrix
    _, eigvecs = eigs(L, k=k, which='SR')
    eigvecs = eigvecs.real

    #Apply k-means clustering on the rows of the eigenvector matrix
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(eigvecs)

    #Assign each node in G to a cluster
    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}

    return clustering

def generate_communities(clusters):
    nb_clusters = len(set(clusters.values()))
    res = [set() for _ in range(nb_clusters)]
    for node,cluster in clusters.items():
        res[cluster - 1].add(node)
    return res

def random_clustering(G, k):
    clustering = {}
    for node in G.nodes():
        clustering[node] = random.randint(1, k)
    return clustering



def main():
    graph_data = nx.read_edgelist("dataset/CA-HepTh.txt", comments="#", delimiter="\t")
    big_connected_component = max(
        nx.connected_components(graph_data), key=len
    )
    subgraph = graph_data.subgraph(big_connected_component)
    print("applying spectral clustering...")
    nb_clusters = 50
    clusters = spectral_clustering(subgraph, k=nb_clusters)
    random_clusters = random_clustering(subgraph, k=nb_clusters)
    print("Clusters:", clusters)
    communities = generate_communities(clusters)
    random_communities = generate_communities(random_clusters)
    print("Random Modularity Estimated: ", nx.algorithms.community.quality.modularity(subgraph, random_communities))
    print("Modularity Estimated: ", nx.algorithms.community.quality.modularity(subgraph, communities))

if __name__ == "__main__":
    main()