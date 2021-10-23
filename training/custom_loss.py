import numpy as np
from sklearn.metrics import f1_score

#lightGBM用のカスタムメトリック
def macro_f1_for_lgb(preds, data):
    """LightGBM の custom metric"""
    y_true = data.get_label()
    y_pred = preds.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    score = f1_score(y_true, y_pred, average='macro')
    return 'macro_f1', score, True

def accuracy_for_lgb(preds, data):
    """精度 (Accuracy) を計算する関数"""
    # 正解ラベル
    y_true = data.get_label()
    # 推論の結果が 1 次元の配列になっているので直す
    N_LABELS = 3  # ラベルの数
    y_pred = preds.reshape(N_LABELS, len(preds) // N_LABELS)
    # 最尤と判断したクラスを選ぶ　
    y_pred = np.argmax(y_pred, axis=0)
    # メトリックを計算する
    acc = np.mean(y_true == y_pred)
    # name, result, is_higher_better
    return 'accuracy', acc, True