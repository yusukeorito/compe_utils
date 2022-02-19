#tabnet baseline
"""
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
"""
import os
import gc
import numpy as np
import pandas as pd
import random
import torch
from collections import defaultdict
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from argparse import Namespace
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error



def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true,y_pred, squared=False)

def rmspe(y_true, y_pred):
    # Function to calculate the root mean squared percentage error
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

config = Namespace(
    INFER=False,
    debug=False,#デバッグモード
    seed=21,
    folds=5,
    workers=4,
    holdout=True,
    num_bins=16,
    #data_path=Path("../input/ubiquant-parquet/"),
)

seed_everything(config.seed)

skf = StratifiedKFold(n_splits=config.folds, shuffle=True)

tabnet_params = dict(
        #cat_idxs=cat_idxs,
        cat_emb_dim=1,
        n_d = 16,
        n_a = 16,
        n_steps = 2,
        gamma =1.4690246460970766,
        n_independent = 9,
        n_shared = 4,
        lambda_sparse = 0,
        optimizer_fn = Adam,
        optimizer_params = dict(lr = (0.024907164557092944)),
        mask_type = "entmax",
        scheduler_params = dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
        scheduler_fn = CosineAnnealingWarmRestarts,
        seed = 42,
        verbose = 10, 
    )  

fit_params = dict(
    max_epochs = 355,
    patience = 50,
    batch_size = 1024*20, 
    virtual_batch_size = 128*20,
    num_workers = 4,
    drop_last = False,
    )

features = [] #特徴量のカラム名

def fit_tabnet(X:pd.DataFrame, y:pd.DataFrame,cv, model_params:dict, fit_params:dict):    
    X['preds'] = -1000
    scores = defaultdict(list)
    features_importance= pd.DataFrame()
    
    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records, ), dtype=np.float)
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"=====================fold: {fold}=====================")  
        #trn_ind, val_ind = X.fold!=fold, X.fold==fold
        print(f"train length: {tr_idx.sum()}, valid length: {va_idx.sum()}")
        X_train=X.loc[tr_idx, features].values
        y_train=y.loc[tr_idx].values.reshape(-1,1)
        X_val=X.loc[va_idx, features].values
        y_val=y.loc[va_idx].values.reshape(-1,1)

        clf =  TabNetRegressor(**tabnet_params)
        clf.fit(
          X_train, y_train,
          eval_set=[(X_val, y_val)],
          **fit_params,
          )
        
        clf.save_model(f'TabNet_seed{config.seed}_{fold}')


        preds = clf.predict(X.loc[va_idx, features].values)
        oof_pred[va_idx] = preds
        X.loc[va_idx, "preds"] = preds
        models.append(clf)
        
        score = rmse(y.loc[va_idx], preds)
        print(f" - fold{fold+1} RMSE - {score:.4f}")
        
        scores["rmse"].append(score)
     
        del X_train,X_val,y_train,y_val
        gc.collect()
        
        
    print(f"TabNet {config.folds} folds mean rmse: {np.mean(scores['rmse'])}")
    X.filter(regex=r"^(?!f_).*").to_csv("preds.csv", index=False)
    return  oof_pred, models, features_importance