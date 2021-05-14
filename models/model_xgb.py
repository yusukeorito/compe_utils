import os
import sys
import pandas as pd
import numpy as np


import xgboost as xgb
from xgboost import XGBModel

#XGBoostのラッパーモデル
class XGBModel:
    def __init__(self, model_params=None, fit_params=None):
        self.model_params = model_params
        self.fit_params = fit_params
        self.model =None
    
    def build_model(self):
        self.model = XGBModel(**self.model_params) #インスタンス化

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.build_model()
        self.model.fit(tr_x, tr_y,eval_set=[[va_x, va_y]],
        **self.fit_params)

    def predict(self, x):
        preds = self.model.predict(x)
        return preds

    def save(self, path):
        Util.dump(self.model, path)

    def load(self, path):
        self.model = Util.load(path)

    def get_saving_path(self, path):
        return path


