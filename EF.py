from base import IClassifier
import numpy as np
from sklearn.decomposition import PCA

class EigenFacesClassifier(IClassifier):
    def __init__(n_components=1, distance=np.linalg.norm(a-b)):
        self._distance = distance
        
        self._n_components = n_components
        self._pca = PCA(self._n_components)
        
        self._mean = None
        self._cov = None
        
        self._data = None
        self._labels = None
    
    
    def _image_to_vector(self, X):
        vector_size = X.shape[1] * X.shape[2]
        vectors = np.zeros((X.shape[0], vector_size))
        for i in range(vectors.shape[0]):
            vectors[i] = X[i].reshape(vector_size)
        return vectors
    
    
    def _get_distance_matrix(vectors):
        distance_matrix = np.zeros((vectors.shape[0], self._data.shape[0]))
        for i in range(vectors.shape[0]):
            for j in range(self._data.shape[0]):
                distance_matrix[i, j] = self._distance(vectors[i], self._data[j])
        return distance_matrix
        
        
    def fit(self, X, Y):   
        data = self._image_to_vector(X)
        self._mean = np.mean(data, axis=0)
        for i in range(data.shape[0]):
            data[i] = data[i] - self._mean
        self._cov = np.cov(data)
        
        self._pca.fit(self._cov)
        
        self._data = self._pca.transform(data)
        self._labels = np.copy(Y)

        
    def predict(self, X):
        vectors = self._pca.transfrom(self._image_to_vector(X))
        distance_matrix = self._get_distance_matrix(vectors)
        return self._labels(np.argmin(distance_matrix, axis=1))
        
    
    def predict_proba(self, X):
        vectors = self._pca.transfrom(self._image_to_vector(X))
        distance_matrix = np.exp(self._get_distance_matrix(vectors) * -1)
        sums = np.sum(distance_matrix, axis=1)
        distance_matrix = distance_matrix / sums
        return self._labels(np.argmax(distance_matrix, axis=1))