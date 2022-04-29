from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)