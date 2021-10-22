import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

#特徴量のヒストグラムを作成する関数
def plot_hist(X, title=None, x_label=None):
    fig, ax = plt.subplots()
    sns.displot(X, kde=False, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    fig.show()


#特徴量と目的変数との散布図を作成する関数
def plot_scatter(X, y, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplot()
    sns.scatterplot(X, y, ci=None, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.show()

#特徴量と目的変数の散布図(複数)
def plot_scatters(X, y, title=None):
    columns = X.columns
    fig, axes = plt.subplots(nrows=None, ncols=None)
    for ax, c in zip(axes.ravel(), columns):
        sns.scatterplot(X[c], y, ci=None, ax=ax, label=None) #ここを変更することで応用できる sns.histplot()など
        
        ax.set_xlabel(c)
        ax.set_ylabel('target')
        ax.legend()

    fig.suptitle(title)
    fig.show()

#train dataとtest dataの差分を確かめる関数

fig, axes = plt.subplots(figsize=(n_rows*3, n_cols*3), nrows=None, ncols=None)
for ax, c in zip(axes.ravel(), columns):
    venn2(
        subsets=(set(train[c].unique()), set(test[c].unique())),
        set_labels=('Train', 'Test'),
        ax=ax
    )

    ax.set_title(c)
fig.tight_layout()