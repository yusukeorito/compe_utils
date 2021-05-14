import os 
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#KNNのラッパーモデル
class KNN:
    def __init__(self, model_params, fit_params):
        self.knn = None
        self.random_state = None
        self.learning_rate = None
        self.model_params = model_params
        self.fit_params = fit_params

    def build_model(self):
        self.knn = KNeighborsClassifier(**self.model_params)

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.build_model()
        
        x_col = tr_x.columns
        knn_col = [k for k in x_col if k.startwith("knn__")]

        tr_knn = tr_x[knn_col]
        va_knn = va_x[knn_col]
        self.knn.fit(tr_knn.values, tr_y.values)

    def predict(self, x):
        x_col = x.columns
        knn_col = [k for k in x_col if k.startwith("knn__")]
        x_knn = x[knn_col]
        preds_knn = self.knn.predict_proba(x_knn.values)
        return preds_knn
        
    def save(self, path):
        models =self.knn
        Util.dump(models, path)
    
    def load(self, path):
        self.knn = Util.load(path)

    def get_saving_path(self, path):
        return path