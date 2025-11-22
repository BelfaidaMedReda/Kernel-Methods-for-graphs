import networkx as nx


def main():
    graph_data = nx.read_edgelist("dataset/CA-HepTh.txt", comments="#", delimiter="\t")
    print("Number of nodes:", graph_data.number_of_nodes())
    print("Number of edges:", graph_data.number_of_edges())
    # Find the largest connected component
    big_connected_component = max(
        nx.connected_components(graph_data), key=len
    )
    subgraph = graph_data.subgraph(big_connected_component)
    print("Number of nodes in the largest connected component:", subgraph.number_of_nodes())
    print("Number of edges in the largest connected component:", subgraph.number_of_edges())

if __name__ == "__main__":
    main()