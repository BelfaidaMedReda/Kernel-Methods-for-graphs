import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import RandomWalk
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab) #add word in vocab 

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab #we end up only with the words we have encountered in the docs

def create_graph_of_words(list_of_docs, vocab, window_size=4):
    res = []  #list of graphs
    left = window_size // 2
    right = window_size - left

    for doc in list_of_docs:
        graph = nx.Graph()
        for word_index, word in  enumerate(doc):
            graph.add_node(vocab[word])
            graph.nodes[vocab[word]]['label'] = word
            left_bound = max(0, word_index - left)
            right_bound = min(len(doc), word_index + right + 1)
            for neighbor_index in range(left_bound, right_bound):
                if neighbor_index != word_index:
                    neighbor_word = doc[neighbor_index]
                    if graph.has_edge(vocab[word], vocab[neighbor_word]):
                        graph[vocab[word]][vocab[neighbor_word]]['weight'] += 1
                    else:
                        graph.add_edge(vocab[word], vocab[neighbor_word], weight=1)
        res.append(graph)

    return res



def main():
    path_to_train_set = 'dataset/train_5500_coarse.label'
    path_to_test_set = 'dataset/TREC_10_coarse.label'

    # Read and pre-process train data
    train_data, y_train = load_file(path_to_train_set)
    train_data = preprocessing(train_data)

    # Read and pre-process test data
    test_data, y_test = load_file(path_to_test_set)
    test_data = preprocessing(test_data)

    # Extract vocabulary
    vocab = get_vocab(train_data, test_data)
    reverse_vocab = {v: k for k, v in vocab.items()}
    print("Vocabulary size: ", len(vocab)) 

    # Create graph of words for train and test data
    G_train = create_graph_of_words(train_data, vocab, window_size=4)
    G_test = create_graph_of_words(test_data, vocab, window_size=4)
    print("Number of training graphs: ", len(G_train))
    print("Number of testing graphs: ", len(G_test))

    # Visualize the first training graph
    # nx.draw(G_train[0], with_labels=True)
    # plt.show()
    
    print("Converting to Grakel format...")
    G_train = graph_from_networkx(G_train, node_labels_tag='label')
    G_test = graph_from_networkx(G_test, node_labels_tag='label')


    # Testing Weisfeiler-Lehman kernel
    wl_kernel = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram)
    K_train = wl_kernel.fit_transform(G_train)
    K_test = wl_kernel.transform(G_test)

    print("Testing Weisfeiler-Lehman Kernel...")
    print("Weisfeiler-Lehman Kernel - Training Kernel Matrix Shape: ", K_train.shape)
    print("Weisfeiler-Lehman Kernel - Testing Kernel Matrix Shape: ", K_test.shape) 

    
    # Training a SVM classifier
    clf = svm.SVC(kernel='precomputed')
    clf.fit(K_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(K_test)
    wl_accuracy = accuracy_score(y_test, y_pred)
    
    print("Weisfeiler-Lehman Kernel SVM Classification Accuracy: ", f"{wl_accuracy * 100:.2f} %")


if __name__ == "__main__":
    main()