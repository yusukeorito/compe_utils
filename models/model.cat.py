#catboost pipeline
import os
import gc
import sys
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import pickle
from sklearn.metrics import roc_auc_score
import catboost as cat

sys.path.append('.')
from training.validation import get_stratifiedkfold, get_groupkfold

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

#回帰
cat_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'num_boost_round': 10000,
    'learning_rate': 0.01,
    'random_state': 42,
    'task_type': 'CPU',
    'depth': 6,
}


#atma cup13 ２値分類
def fit_catboost(X, y, params, folds, categorycal_list=[], add_suffix=''):
    """
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'num_boost_round': 10000,
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'depth': 6,
    }
    """
    oof_pred = np.zeros(len(y), dtype=np.float32)

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
        model = cat.CatBoostClassifier(**params) #分類
        # model = cat.CatBoostRegressor(**params) 回帰
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
        #pred_i = model.predict(x_valid)[:] 回帰
        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        #score = round(RMSE(y_valid, pred_i), 5) 回帰
        print(f'Performance of the prediction: {score}\n')

    score = round(roc_auc_score(y, oof_pred), 5)
    #score = round(RMSE(y, oof_pred), 5) 回帰
    print(f'All Performance of the prediction: {score}')
    del model
    gc.collect()
    return oof_pred

def pred_catboost(X, data_dir: Path, categorical_list,  add_suffix=''):
    models = glob(str(data_dir / f'cat*{add_suffix}.pkl'))
    models = [pickle.load(open(model, 'rb')) for model in models]
    X = cat.Pool(X, cat_features=categorical_list)
    preds = np.array([model.predict_proba(X)[:, 1] for model in models])
    preds = np.mean(preds, axis=0)
    return preds

if __name__ == '__main__':
    train = pd.read_csv('')
    test = pd.read_csv('')
    train_X = train.drop(columns=['target'])
    test_X = test.copy()
    categorical_list = [] 



    final_catoof = []
    final_catsub = []
    for fold in [5]:
        folds = get_stratifiedkfold(train, 'target',  fold, 2022)
        cat_oof = fit_catboost(train_X, train['target'], cat_params, folds, [], f'_cat_numfolds{fold}')
        cat_sub = pred_catboost = pred_catboost(test_X, Path(''), [], f'_cat_numfolds{fold}')
        final_catoof.append(cat_oof)
        final_catsub.append(cat_sub)

    sub = pd.read_csv('')
    sub['target'] = final_catsub[0]
    sub.to_csv('submission.csv', index=None)
