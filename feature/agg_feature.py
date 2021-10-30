import pandas as pd
import numpy as np
import typing as tp
from itertools import product

#集約特徴量作成関数
def create_agg_features(
    df:pd.DataFrame,
    group_ids: tp.Sequence,
    feature_names: tp.List[str],
    agg_names: tp.List[str]
) -> pd.DataFrame:
    """
    pandas.core.groupby.DataFrameGroupBy.aggregate で集約特徴を作成する関数

    Args:
        df (pd.DataFrame)            : 特徴量の集約元となるデータ
        group_ids (Sequence)         : 集約する group を示す id のシーケンス
        feature_names (List[str])    : 集約の対象となるカラムのリスト
        aggregation_names (List[str]): 集約の操作のリスト ["sum","mean", ...]

    Returns:
        agg_feat (pd.DataFrame): 集約した特徴. index は group_id となっている.
    """

    # # pandas.core.groupby.DataFrameGroupBy.aggregate に渡す辞書を作成
    agg_dict = {}
    for f_name, a_name, in product(feature_names, agg_names):
        agg_dict[f'{f_name}_{a_name}'] = pd.NamedAgg(columns=f_name, aggfunc=a_name)
    
    agg_feat = df.groupby(group_ids).agg(**agg_dict) #df.groupby(group_ids)[fe_name].agg(a_name)
    return agg_feat

#同一グループ内での順序特徴量作成関数
def create_order_features(
    df:pd.DataFrame,
    group_ids:tp.List[str],
    feature_names:tp.List[str],
    prefix:str=None,
    suffix:str=None,
    is_ascending:bool=True,
    rank_method: str='average'
) -> pd.DataFrame:

    """
    group 内での順序特徴を作成.

    Args:
        df (pd.DataFrame)        : 順序の算出元となるデータ
        group_ids (Sequence)     : 集約する group を示す id のシーケンス
        feature_names (List[str]): 順序を算出する対象となるカラムのリスト
        prefix (str)             : 特徴量名の先頭に付加する文字列, Default: ""
        suffix (str)             : 特徴量名の末尾に付加する文字列, Default: ""
        is_ascending (bool)      : {True(昇順), False(降順)}, Default: True
        rank_method (str)        : 順序の算出方法. 以下より選択
            {‘average’, ‘min’, ‘max’, ‘first’, ‘dense’}, Default: ‘average’

    Returns:
        rank_feat (pd.DataFrame) : 算出した順序特徴
    """
    rank_feat = df[feature_names].groupby([group_ids]).rank(ascending=is_ascending,method=rank_method)

    if prefix != None:
        rank_feat.add_prefix(f"{prefix}_")
    if suffix != None:
        rank_feat.add_suffix(f"_{suffix}")

    return rank_feat



def create_subdf_features(df:pd.DataFrame) -> pd.DataFrame:

    agg_method_names = ["sum", "min", "max", "mean", "median", "std",]

    #上で作成した関数を中で利用する
    agg_A = create_agg_features(df, df['id'], ['left', 'right','top', 'bottom'],
    ['sum','mean','max','min','median','std'])

    agg_B = create_agg_features(df, df['id'], ['width', 'height','size', 'ratio'],
    ['sum','mean','max','min','median','std'])

    output_df = pd.concat([agg_A, agg_B], axis=1)
    #入力のdfにmergeして返したい時などはここに書き加える

    return output_df

def create_subdf_order_features(df:pd.DataFrame) -> pd.DataFrame:

    #同じ集計に対して昇順と降順の両方で作成する
    pos_rank_asc = create_order_features(df, df['id'], ['left', 'right','top', 'bottom'],
    prefix='box', suffix='asc_order', is_ascending=True, rank_method='dense')

    pos_rank_dsc = create_order_features(df, df['id'], ['left', 'right','top', 'bottom'],
    prefix='box', suffix='asc_order', is_ascending=False, rank_method='dense')

    output_df = pd.concat([pos_rank_asc, pos_rank_dsc], axis=1)

    return output_df
