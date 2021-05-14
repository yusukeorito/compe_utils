#特徴量を生成するクラス
import pandas as pd
import numpy as np
from pandas.core.arrays import categorical


#抽象クラス
class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


#Numerical_feature(変換なし)
class NumericBlock(AbstractBaseBlock):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        use_columns = [

        ]
        return input_df[use_columns].copy()
    
#Categorical_features
"""
カテゴリ特徴量はtrain_data, test_dataをfit->transformの構造によって変形しなくてはいけないので以下の
read_whole_dfという関数を定義する
"""

def read_whole_df(train, test):
    return pd.concat([train, test], axis=0, ignore_index=True)

#CountEncodingFeature
class CountEncodingBlock(AbstractBaseBlock):
    def __init__(self, column: str):
        self.column = column

    def fit(self, input_df, y=None):
        master_df = read_whole_df()
        vc = master_df[self.column].value_counts()
        self.count_ = vc
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.column] = input_df[self.column].map(self.count_)
        return output_df.add_prefix('CE_')

#OneHotEncoding_feature
class OneHotEncodingBlock(AbstractBaseBlock):
    def __init__(self, column, min_count=None):
        self.columns = column 
        self.min_count = min_count

    def fit(self, input_df, y=None):
        x = input_df[self.column]
        vc = x.value_counts() #vcはuniqueがインデックスになったseriesで返ってくる
        categories = vc[vc > self.min_count].index #min_count以上のuniqueな値がpandasのindexで返ってくる
        self.categories_ = categories
        return self.transform(input_df)

    def transform(self, input_df):
        x = input_df[self.column]
        cat = pd.Categorical(x, categories=self.categories_)
        output_df = pd.get_dummies(cat)
        output_df.columns = output_df.columns.tolist()
        return output_df.add_prefix(f'{self.column}=')

#TargetEncoding_feature
class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, use_columns: List[str], cv):#target encodingはCVが引数に必要である。
        super(TargetEncodingBlock, self).__init__()

        self.mapping_df_ None
        self.use_columns = use_columns
        self.cv = list(cv)
        self.n_fold = len(cv)
    
    def create_mapping(self, input_df, y):
        self.mapping_df_ = {}
        self.y_mean_ = np.mean(y)

        output_df = pd.DataFrame()
        target = pd.Series(y)

        for col_name in self.use_columns:
            keys = input_df[col_name].unique()#keysはユニークな要素のnp.array
            X = input_df[col_name]

            oof = np.zeros_like(X, dtype=np.float)#target　encodingによって生成される特徴量の入れ物

            for tr_idx, va_idx in self.cv:
                _df = target[tr_idx].groupby(X[tr_idx]).mean()   
                _df = _df.reindex(keys)
                _df = _df.fillna(_df.mean())
                oof[va_idx] = input_df[col_name][va_idx].map(_df.to_dict())

            output_df[col_name] = oof

            self.mapping_df_[col_name] = target.groupby(X).mean()
        
        return output_df
    
    def fit(self, input_df: pd.DataFrame, y=None, **kwrgs) -> pd.DataFrame:
        output_df = self.create_mapping(input_df, y=y)
        return output_df.add_prefix('TE_')
    
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()

        for c in self.use_columns:
            output_df[c] = input_df[c].map(self.mapping_df_[c]).fillna(self.y_mean_)
        
        return output_df.add_prefix('TE_')

    """
    _dfの様子
    
    Embarked
    C    0.544828
    Q    0.377049
    S    0.349112
                
    TargetEncodingではfit transformの構造をもち、内部状態が保存される。そのためtargetをもたないtest_dfは
    fitにより保存されたtrainのtargetの内部情報をもとに変換される.
    """
