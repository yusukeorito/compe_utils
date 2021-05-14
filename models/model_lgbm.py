from lightgbm import LGBMModel

#LightGBMのラッパーモデル
class LGBM:
    def __init__(self, model_params, fit_params):
        self.lgb = None
        self.columns = None
        self.feature_importance = None
        self.random_state = None
        self.model_params = model_params
        self.fit_params = fit_params

    def build_model(self):
        self.lgb = LGBMModel(**self.model_params)

    def fit(self, tr_x, tr_y, va_x=None, va_y=None):
        self.build_model()
        x_col = tr_x.columns
        lgb_col = [k for k in x_col if k.startswith("lgb__")]
        tr_lgb = tr_x[lgb_col]
        va_lgb = va_x[lgb_col]

        self.columns = lgb_col

        self.lgb.fit(tr_lgb.values, tr_y.values,
                     eval_set=[[va_lgb.values, va_y.values]],
                     early_stopping_rounds=100,
                     verbose=0)

        self.feature_importances_ = self.lgb.feature_importances_

    def predict(self, x):
        x_col = x.columns
        lgb_col = [k for k in x_col if k.startwith("lgb__")]

        x_lgb = x[lgb_col]
        preds_lgb = self.lgb.predict(x_lgb.values)

        return preds_lgb

    def save(self, path):
        models = self.lgb
        Util.dump(models, path)
    
    def load(self, path):
        self.lgb = Util.load(path)

    def get_saving_path(self, path):
        return path