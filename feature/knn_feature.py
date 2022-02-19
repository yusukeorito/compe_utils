#K近傍法を用いた特徴量作成
import numpy as np
from numpy import ndarray
from scipy.spatial import distance
from sklearn.neighbors import  KNeighborsClassifier

class KNNFeatureExtractor:
    def __init__(self, n_neighbors=5):
        self.knn = KNeighborsClassifier(n_neighbors+1)

    def fit(self, X,y):
        self.knn.fit(X,y)
        self.y = y if isinstance(y, np.ndarray) else np.array(y)
        return self

    def transform(self, X, is_train_data):
        distance, indexes = self.knn.kneighbors(X) 
        """
        args
        近傍点のインスタンスのインデックスと近郷点までの距離を返す
        np.array(len(X), 近傍数)
        """
        
