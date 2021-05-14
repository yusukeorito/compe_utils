import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#GBDTのfeature importanceの出力と可視化
folds = 5
def tree_importance(X, y, model, model_params, fit_params, cv, cv_key, folds):
    est = model(model_params, fit_params)
    feature_importance_df = pd.DataFrame() #dataframeの作成
    for i, (tr_idx, va_idx) in enumerate(cv(X, y, n_splits=folds, cv_key=cv_key)):
        tr_x, va_x = X.values[tr_idx], X.values[va_idx] #dataframeではなくarrayで返す
        tr_y, va_y = y.values[tr_idx], y.values[va_idx]

        est.fit(tr_x, tr_y, va_x, va_y)
        _df = pd.DataFrame()
        _df['feature_importance'] = est.model.feature_importances_
        _df['column'] = X.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True) #fold分結合する
    order = feature_importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
    sns.boxplot(data=feature_importance_df, y='columns', x='feature_importance', order=order,
    ax=ax, palette='viridis')
    fig.title_layout()
    ax.grid()
    ax.set_title('feature_importance')
    fig.tight_layout()
    return fig, feature_importance_df
        