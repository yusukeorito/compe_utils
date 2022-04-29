#lightgbm pipeline(columnsさん)
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

import lightgbm as lgbm
sys.path.append('.')
from training.validation import get_stratifiedkfold, get_groupkfold


#二値分類
lgbm_params = {
 'objective': 'binary',
 'metric': 'AUC',
 'boosting': 'gbdt',
 'learning_rate': 0.01,
 'feature_pre_filter': False,
 'lambda_l1': 2e-05,
 'lambda_l2': 3e-06,
 'num_leaves': 2**8-1,
 'feature_fraction': 0.6,
 'bagging_fraction': 1.0,
 'bagging_freq': 0,
 'min_child_samples': 20,
 'num_iterations': 10000,
 'early_stopping_round': 100,
 # 'device': 'gpu',
}

def fit_lightgbm(X, y, params, folds, add_suffix=''):
    """
    lgbm_params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
    }
    """
    models = []
    oof_pred = np.zeros(len(y), dtype=np.float64)

    fold_unique = sorted(folds.unique())
    for fold in fold_unique:
        idx_train = (folds!=fold)
        idx_valid = (folds==fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        lgbm_train = lgbm.Dataset(x_train, y_train)
        lgbm_valid = lgbm.Dataset(x_valid, y_valid, reference=lgbm_train)
        model = lgbm.train(
            params=params,
            train_set=lgbm_train,
            valid_sets=[lgbm_train, lgbm_valid],
            num_boost_round=10000,
            early_stopping_rounds=50,
            verbose_eval=100,
        )
        pickle.dump(model, open(os.path.join(TRAINED+f'/lgbm_fold{fold}{add_suffix}.pkl'), 'wb'))
        models.append(model)
        pred_i = model.predict(x_valid, num_iteration=model.best_iteration)
        oof_pred[x_valid.index] = pred_i
        score = round(RMSE(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}')

    score = round(RMSE(y, oof_pred), 5)
    print(f'All Performance of the prediction: {score}')
    del model
    gc.collect()
    return oof_pred, models

def pred_lightgbm(X, data_dir: Path, add_suffix=''):
    models = glob(str(data_dir / f'lgbm*{add_suffix}.pkl'))
    models = [pickle.load(open(model, 'rb')) for model in models]
    preds = np.array([model.predict(X) for model in models])
    preds = np.mean(preds, axis=0)
    return preds

def visualize_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.

    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        # _df["feature_importance"] = model.feature_importances_
        _df["feature_importance"] = model.feature_importance(importance_type="gain")
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column")\
        .sum()[["feature_importance"]]\
        .sort_values("feature_importance", ascending=False).index[:50]#上位50個を表示

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x="feature_importance", 
                  y="column", 
                  order=order, 
                  ax=ax, 
                  palette="viridis", 
                  orient="h")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    return fig, ax


def pred_lightgbm(X, data_dir: Path, add_suffix=''):
    models = glob(str(data_dir / f'lgbm*{add_suffix}.pkl'))
    models = [pickle.load(open(model, 'rb')) for model in models]
    preds = np.array([model.predict(X) for model in models])
    preds = np.mean(preds, axis=0)
    return preds

if __name__ == '__main__':
    train = pd.read_csv('')
    test = pd.read_csv('')
    train_X = train.drop(columns=['target'])
    test_X = test.copy()

    final_lgbmoof = []
    final_lgbmsub = []
    for fold in [5]:
        folds = get_groupkfold(train, 'target', 'date', fold)
        lgbm_oof = fit_lightgbm(train_X, train['target'], lgbm_params, folds, f'_lgbm_numfolds{fold}')
        lgbm_sub = pred_lightgbm(test_X, Path(''), f'_lgbm_numfolds{fold}')
        final_lgbmoof.append(lgbm_oof)
        final_lgbmsub.append(lgbm_sub)


