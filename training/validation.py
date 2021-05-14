import os
import pandas as pd
import numpy as np
from logging import Logger
from scipy.sparse.construct import random
from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold

#KFold
def kf(train_x, train_y, n_splits, random_state=2021, key=None):
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    return list(kf.split(train_x, train_y, key))
    

#StratifiedKFold
def skf(train_x, train_y, n_splits, random_state=2021, key=None):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    return list(skf.split(train_x, train_y, key))


#training
def train_cv(X, y, model,
            cv, cv_key,
            folds, seeds,
            metrics, name=""):

    est = model()
    
    oof_seeds = []
    for seed in seeds: #seedでaverageが取れるようになっている
        oof, va_idxes = [], []
        train_x, train_y = X.copy(), y.copy()
        fold_idx = cv(train_x, train_x, n_splits=folds, random_state=seed, key=cv_key)

        if est.random_state is not None:
            est.random_state = seed

        for cv_num, (tr_idxm va_idx) in enumerate(fold_idx):
            tr_x, va_x = train_x.iloc[tr_x].reset_index(drop=True), train_x.iloc[va_x].reset_index(drop=True)
            tr_y, va_y = train_y.iloc[tr_x].reset_index(drop=True), train_y.iloc[va_x].reset_index(drop=True)
            va_idxes.append(va_idx)

            model_name = os.path.join(TRAINED, f"{name}_SEED{seed}_FOLD{cv_num}_model.pkl")
            if os.path.isfile(est.get_saving_path(model_name)):
                est.load(model_name)
            else:
                est.fit(tr_x, tr_y, va_x, va_y)
                est.save(model_name)
            
            pred = est.predict(va_x)
            oof.append(pred)

            score = matrics(va_y, pred)
            logger.info(f'SEED{seed}, FOLD:{cv_num} >>>> val_score:{score:.4f}')

        va_idxes = np.concatenate(va_idxes)
        oof = np.concatenate(oof)
        order = np.argsort(va_idxes)
        oof = oof[order]
        oof_seeds.append(oof)
        logger.info(f'seed:{seed} score:{metric(train_y, oof):.4f}\n')
    
    oof = np.mean(oof_seeds, axis=0)
    Util.dump(oof, f'{PREDS}/oof.pkl')
    logger.info(f'model:{name} score:{metrics(train_y, oof):.4f}\n')

    return oof




        