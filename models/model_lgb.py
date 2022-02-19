#LightGBM
import sys
import pandas as pd
import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.append('.')
from Utils.make_folder import REPORTS, PREDS, TRAINED
from Utils.timer import Timer
from Utils.Log import Util


#LightGBMによる学習
def fit_lgb(
    X:pd.DataFrame,
    y:pd.DataFrame,
    cv,
    model_params:dict,
    fit_params:dict,
    fobj=None, feval=None,):

    X, y = X.values, y.values

    """LightGBMをcross validationで学習"""
    models = []
    n_records = len(X)
    n_labels = len(np.unique(y))

    oof_pred = np.zeros((n_records, n_labels), dtype=np.float)

    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        tr_X, tr_y = X[tr_idx], y[tr_idx]
        va_X, va_y = X[va_idx], y[va_idx]

        train_data = lgb.Dataset(tr_X, label=tr_y)
        valid_data = lgb.Dataset(va_X, label=va_y)

        with Timer(prefix="fit fold={}".format(i)):
            clf = lgb.train(
                model_params,
                train_data,
                **fit_params,
                valid_names=['train', 'valid'],
                valid_sets=[train_data, valid_data],
                feval=feval #自作の損失関数を適用
            )

        pred_i = clf.predict(va_X, num_iteration=clf.best_iteration)
        oof_pred[va_idx] = pred_i
        models.append(clf)

        y_pred_label = np.argmax(pred_i, axis=1)

        score = f1_score(va_y, y_pred_label, average='macro')
        print(f" - fold{i+1} F1score - {score:.4f}")

    oof_label = np.argmax(oof_pred, axis=1)
    score = f1_score(y, oof_label, average='macro')

    print(f"{score:.4f}")
    
    Util.dump(oof_pred, f"{PREDS}/oof.pkl") #oofの保存
    
    return oof_pred, models



model_params = {
    "boosting_type": "gbdt",

    "num_class": 9, #タスクによって変更する
    "objective": "multiclass", #タスクによって変更する
    "metric": "None", #custom lossを使うときはここをnoneにしておく

    "learning_rate": 0.05,
    "max_depth": 12,

    "reg_lambda": 1.,
    "reg_alpha": .1,

    "colsample_bytree": .5,
    "min_child_samples": 10,
    "subsample_freq": 3,
    "subsample": .8,

    "random_state": 2021,
    "verbose": -1,
    "n_jobs": 8,
}

fit_params = {
    "num_boost_round": 20000,
    "early_stopping_rounds": 200,
    "verbose_eval": 100,
}

#特徴量重要度を可視化する関数
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
    

    
#混合行列を描画する関数　分類問題の結果の可視化に便利
#確率ではなくラベルで渡す
def visualize_confusion_matrix(
    y_true, pred_label,
    ax: tp.Optional[plt.Axes] = None,
    labels: tp.Optional[list] = None,
    conf_options: tp.Optional[dict] = None,
    plot_options: tp.Optional[dict] = None
) -> tp.Tuple[plt.Axes, np.ndarray]:
    """
    visualize confusion matrix
    Args:
        y_true:
            True Label. shape = (n_samples, )
        pred_label:
            Prediction Label. shape = (n_samples, )
        ax:
            matplotlib.pyplot.Axes object.
        labels:
            plot labels
        conf_options:
            option kwrgs when calculate confusion matrix.
            pass to `confusion_matrix` (defined at scikit-learn)
        plot_options:
            option key-words when plot seaborn heatmap
    Returns:
    """

    _conf_options = {
        "normalize": "true",
    }
    if conf_options is not None:
        _conf_options.update(conf_options)

    _plot_options = {
        "cmap": "Blues",
        "annot": True
    }
    if plot_options is not None:
        _plot_options.update(plot_options)

    conf = confusion_matrix(y_true=y_true,
                            y_pred=pred_label,
                            **_conf_options)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, ax=ax, **_plot_options)
    ax.set_ylabel("Label")
    ax.set_xlabel("Predict")

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params("y", labelrotation=0)
        ax.tick_params("x", labelrotation=90)

    return ax, conf

#_ = visualize_confusion_matrix(y, np.argmax(oof, axis=1), conf_options={ "normalize": None }, plot_options={ "fmt": "4d" })

