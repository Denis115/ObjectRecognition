from base import IClassifier
import numpy as np
from sklearn.decomposition import PCA

class EigenFacesClassifier(IClassifier):
    def __init__(n_componenets=1):
        self._n_components = n_componenets
        self._mean = None
        self._cov = None
        self.eigenvalues = None
        self.eigenvectors = None
    
    def _image_to_vector(self, X):
        vector_size = X.shape[1] * X.shape[2]
        vectors = np.zeros((X.shape[0], vector_size))
        for i in range(vectors.shape[0]):
            vectors[i] = X[i].reshape(vector_size)
        return vectors.T
        
        
    def fit(self, X, Y):
        
        data = self._image_to_vector(X)
        self._mean = np.mean(data, axis=1)
        for i in range(data.shape[1]):
            data[:,i] = data[:,i] - self._mean
        self._cov = np.cov(data)
        
        pca = PCA(self._n_components)
        pca.fit(data)
        self.eigenvectors = pca.components_
        self.eigenvalues = pca.explained_ratio_
        
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass