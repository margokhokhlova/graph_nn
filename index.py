import faiss
import os
import numpy as np


class BagOfNodesIndex:
    """
    Index object for storing a database of node embeddings and performing nearest neighbour search.
            Params
    k: number of nearest neighbours to consider when matching a node with database nodes.
            Methods
    train: performs KMeans clustering and quantization to build an inverted index.
    save: saves the index to disk
    load: loads the index from disk
    search: gets closest neighbours to input list of descriptors
    """
    def __init__(self, dimension=40):
        coarse_quantizer = faiss.IndexFlatL2(dimension)
        N_CENTROIDS = 256 # Size of the vocabulary
        CODE_SIZE = 8 # Number of subquantizers, dimension % CODE_SIZE should be zero
        N_BITS = 4 # Bits per subquantizer
        self.index = faiss.IndexIVFPQ(coarse_quantizer, dimension, N_CENTROIDS, CODE_SIZE, N_BITS)
        self.index_ownership = []
        self.k = 3
        self.size = 0

    def train(self, features, ownership):
        """
                Inputs:
        features (ndarray/list): NxD array or list of node embeddings (features)
        ownership (ndarray/list): N elements array or list linking each feature to its source image ID
        """

        index_features = np.array(features).astype('float32')
        self.index_ownership = np.array(ownership)

        self.index.train(index_features)
        self.index.add(index_features)

        self.size = index_features.shape[0]

    def search(self, query_features):
        """
                Inputs:
        query_features (ndarray/list): NxD array or list of query node embeddings (features)
        """

        distances, matched_descriptors = self.index.search(query_features.astype('float32'), self.k)

        corresponding_images = self.index_ownership[matched_descriptors]

        sumofinliers = np.zeros(self.size)

        candidate_images, nboccurences = np.unique(corresponding_images, return_counts=True)
        for i, candidate in enumerate(candidate_images):
            sumofinliers[candidate] = nboccurences[i]

        winners = np.argsort(sumofinliers)[::-1].astype(int)
        return winners, sumofinliers

    def save(self, path):
        """
        Saves index and ownership array to disk
                Inputs:
        path (str): destination folder
        """
        faiss.write_index(self.index, os.path.join(path, "ivfpq.index"))
        np.save(os.path.join(path, "ownership.npy"))

    def load(self, path):
        """
        Loads index and ownership array from disk
                Inputs:
        path (str): destination folder
        """
        self.index = faiss.read_index(os.path.join(path, "ivfpq.index"))
        self.index_ownership = np.load(os.path.join(path, "ownership.npy"))


if __name__ == '__main__':
    index = BagOfNodesIndex()
    features = np.random.randn(10000, 40)
    ownership = np.arange(0, 10000)
    index.train(features, ownership)

    query_features = np.random.randn(15, 40)
    print(index.search(query_features))