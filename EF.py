from base import IClassifier
import numpy as np
from sklearn.decomposition import PCA

class EigenFacesClassifier(IClassifier):
    def __init__(self, n_components=1, distance=lambda x, y: np.linalg.norm(x-y)):
        self._distance = distance
        
        self._n_components = n_components
        self._pca = PCA(self._n_components)
        
        self._mean = None
        self._cov = None
        
        self._data = None
        self._labels = None
        
        self._weights = None
    
    
    def _image_to_vector(self, X):
        # РОЗГОРТКА ЗОБРАЖЕННЯ У ВЕКТОР
        
        vector_size = X.shape[1] * X.shape[2]
        vectors = np.zeros((X.shape[0], vector_size))
        for i in range(vectors.shape[0]):
            vectors[i] = X[i].reshape(vector_size)
        return vectors
    
    
    def _get_distance_matrix(self, vectors):
        # ОБЧИСЛЕННЯ МАТРИЦІ ВІДСТАНЕЙ
        
        distance_matrix = np.zeros((vectors.shape[0], self._weights.shape[0]))
        for i in range(vectors.shape[0]):
            for j in range(self._weights.shape[0]):
                distance_matrix[i, j] = self._distance(vectors[i], self._weights[j])
        return distance_matrix
        
        
    def fit(self, X, Y):   
        # ТРЕНУВАННЯ МОДЕЛІ
        
        # розгортка і видалення усередненого вектору
        data = self._image_to_vector(X)
        self._mean = np.mean(data, axis=0)
        for i in range(data.shape[0]):
            data[i] = data[i] - self._mean
        
        # обчислення коваріаційної матриці та її власних векторів
        self._cov = np.cov(data)    
        self._pca.fit(self._cov)
        
        # проектування у новий простір
        self._data = self._pca.transform(data.T)
        self._labels = np.copy(Y)
        
        # обчислення вагів
        self._weights = data @ self._data


    def predict(self, X):
        # ПЕРЕДБАЧЕННЯ
        
        # розгортка і видалення усередненого вектору
        vectors = self._image_to_vector(X)
        for i in range(vectors.shape[0]):
            vectors[i] = vectors[i] - self._mean
        
        # обчислення вагів
        vectors = vectors @ self._data
        
        # обчислення матриці відстаней
        distance_matrix = self._get_distance_matrix(vectors)
        
        # пошук мінімальних відстаней
        predictions = [self._labels[i] for i in np.argmin(distance_matrix, axis=1)]
        return predictions


    def predict_proba(self, X):
        # ПЕРЕДБАЧЕННЯ У ВИГЛЯДІ ЙМОВІРНОСТЕЙ
        
        # розгортка і видалення усередненого вектору
        vectors = self._image_to_vector(X)
        for i in range(vectors.shape[0]):
            vectors[i] = vectors[i] - self._mean
        
        # обчислення вагів
        vectors = vectors @ self._data
        
        # обчислення матриці відстаней
        distance_matrix = self._get_distance_matrix(vectors)
        
        # перехід у інтервал [0;1] за допомогою softmax
        distance_matrix = np.exp(distance_matrix * -1)
        sums = np.sum(distance_matrix, axis=1)
        distance_matrix = distance_matrix / sums
        
        # пошук максимальних ймовірностей
        return self._labels(np.argmax(distance_matrix, axis=1))