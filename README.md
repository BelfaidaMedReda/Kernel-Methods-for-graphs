# Kernel Methods for Graph (GraphML)

Brief summary
------------
This practical explores text classification via graph-of-words representations and graph classification using graph kernels. The project is organised in three parts (part1, part2, part3). 

Key ideas implemented so far:

- Convert text documents to graph-of-words (nodes = words, edges = co-occurrence within sliding window).
- Build vocabulary, preprocess (tokenization + Porter stemmer), and construct weighted NetworkX graphs per document.
- Apply graph kernels (Weisfeiler–Lehman + VertexHistogram, RandomWalk, custom Graphlet and Shortest-Path kernels) and classify with an SVM using precomputed kernels.
- Experiment with MUTAG (graph dataset) in part3 using torch_geometric -> NetworkX conversion.

Repository structure
--------------------
- part1/main.py
- part2/main.py
- part3/
  - main.py                # Graphlet + shortest-path kernel on MUTAG (torch_geometric -> NetworkX)
  - graph_classification.py# Text -> graph-of-words, Grakel kernels, SVM pipeline
- dataset/
  - train_5500_coarse.label  # training labels+queries (TREC-style: LABEL:query)
  - TREC_10_coarse.label     # test set

What each script does
---------------------
- graph_classification.py
  - load_file: reads LABEL:sentence lines and returns docs and labels.
  - preprocessing: cleans, tokenizes, lowercases and stems tokens (PorterStemmer).
  - get_vocab: builds vocabulary across train+test.
  - create_graph_of_words: for each document, builds a weighted NetworkX graph where nodes are vocab ids and edges count local co-occurrences (sliding window).
  - (later) converts NetworkX graphs to Grakel format and computes kernel matrices (Weisfeiler-Lehman, VertexHistogram, RandomWalk), then trains/evaluates an SVM.

- part3/main.py
  - Uses torch_geometric TUDataset('MUTAG') and converts Data -> NetworkX (to_networkx).
  - Implements:
    - graphlet_kernel: enumerates 3-node graphlets and builds kernel via inner product of graphlet count vectors.
    - shortest_path_kernel: counts shortest-path lengths (up to l_max) and builds kernels from these counts.
  - Trains SVM with kernel='precomputed' and prints accuracy.

Common issues encountered & fixes
-------------------------------
- Warnings still printed after warnings.filterwarnings('ignore'):
  - Place `import warnings; warnings.filterwarnings("ignore")` at the top of the file (before other imports), or run with `PYTHONWARNINGS=ignore` or `python -W ignore script.py`.
  - Some outputs are not Python warnings (they may be logged or printed by C extensions); for those adjust `logging` or redirect stderr.

- torch_geometric / TUDataset usage:
  - TUDataset yields Data objects via iteration. Do not use `dataset.y` directly as a plain list — extract labels via iteration: `y = [int(d.y.item()) for d in dataset]` or build graphs and labels by iterating dataset once.
  - Example: 
    - graphs = [to_networkx(data, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr']) for data in dataset]
    - y = [int(data.y.item()) for data in dataset]

- Grakel conversion errors:
  - Grakel expects simple scalar node labels (hashable). `to_networkx` often stores node features as tensors (e.g. `'x'`) which are not valid node labels for Grakel.
  - Ensure each NetworkX node has a simple scalar/string label before conversion, e.g.:
    - G.nodes[n]['label'] = int(tensor.item()) or str(value)
  - Then use `graph_from_networkx(nx_graphs, node_labels_tag='label')`.

How to run
----------

- Install typical dependencies:
  - pip3 install numpy networkx scikit-learn matplotlib nltk grakel
  - For torch & torch_geometric follow official install instructions for your CUDA / CPU: https://pytorch.org and https://pytorch-geometric.readthedocs.io
  - Download NLTK data if necessary:
    - python3 -c "import nltk; nltk.download('punkt')"

- Run text -> graph pipeline (graph of words + kernels):
  - python3 part3/graph_classification.py

- Run MUTAG experiments (graphlet / shortest path kernels):
  - python3 part3/main.py


