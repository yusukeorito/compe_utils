#特徴量を生成するクラス
import pandas as pd
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


#CountEncoding_feature
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

