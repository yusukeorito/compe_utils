#catboost pipeline
import os
import gc
import pandas as pd
import numpy as np
import matplotlib
import pickle
from sklearn.metrics import roc_auc_score
import catboost as cat


"""
X:pd.DataFrame
y:pd.DataFrame
folds cross validation
params: dice (model_params)
fit_paramsは関数内で指定
"""
#二値分類
cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'num_boost_round': 10000,
    'learning_rate': 0.01,
    'random_state': 42,
    'task_type': 'CPU',
    'depth': 6,
}

#atma cup13 ２値分類
def fit_catboost(X, y, folds, params, categorycal_list=[],add_suffix=''):
  
    oof_pred = np.zeros(len(y), dtype=np.float32)#multiclassの時はクラス数に変更
    models = []
    fold_unique = sorted(folds.unique())

    for fold in fold_unique:
        idx_train = (folds!=fold)
        idx_valid = (folds==fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        cat_train = cat.Pool(
            x_train, 
            label=y_train,
            cat_features=categorycal_list,
        )
        cat_valid = cat.Pool(
            x_valid, 
            label=y_valid,
            cat_features=categorycal_list,
        )
        model = cat.CatBoostClassifier(**params)
        model.fit(
            cat_train,
            early_stopping_rounds=100,
            plot=False,
            use_best_model=True,
            eval_set=[cat_valid],
            verbose=100
        )
        pickle.dump(model, open(f'cat_fold{fold}{add_suffix}.pkl', 'wb'))
        pred_i = model.predict_proba(x_valid)[:, 1]
        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}\n')

        models.append(model)

    score = round(roc_auc_score(y, oof_pred), 5)
    print(f'All Performance of the prediction: {score}')
    #del model
    gc.collect()
    return oof_pred

def pred_catboost(X, models, add_suffix=''):
    #models = glob(str(data_dir / f'cat*{add_suffix}.pkl'))
    #models = [pickle.load(open(model, 'rb')) for model in models]
    preds = np.array([model.predict_proba(X)[:, 1] for model in models])
    preds = np.mean(preds, axis=0)
    return preds