#質的変数と量的変数の相関比を計算する
import pandas as pd
import numpy as np

def correlation_ratio(cat_key, num_key, data):

    categorical=data[cat_key]
    numerical=data[num_key]

    mean=numerical.dropna().mean()
    all_var=((numerical-mean)**2).sum()  #全体の偏差の平方和

    unique_cat=pd.Series(categorical.unique())
    unique_cat=list(unique_cat.dropna())

    categorical_num=[numerical[categorical==cat] for cat in unique_cat]
    categorical_var=[len(x.dropna())*(x.dropna().mean()-mean)**2 for x in categorical_num]  
    #カテゴリ件数×（カテゴリの平均-全体の平均）^2

    r=sum(categorical_var)/all_var

    return r