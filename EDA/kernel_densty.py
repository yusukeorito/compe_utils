#カーネル密度推定を用いてtest_dataのtrainらしさを推定してみる
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

train = pd.DataFrame()
test = pd.DataFrame()

X = train[['x_center_norm','y_center_norm']].values()
clf = KernelDensity(kernel="gausian",bandwidth=0.1) #bandwidthはカーネルの横幅
clf.fit(X) #基本的にモデルの入力はnp.arrayで入力する


xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
zz = clf.score_samples(np.array([xx.reshape(-1,), yy.reshape(-1, )]).T)
fig, axes = plt.subplots(figsize=(20, 10), ncols=2, sharex=True, sharey=True)

ax = axes[0]
ax.scatter(train["x_center_norm"], train["y_center_norm"], c="red", s=5, alpha=.5)
ax.scatter(test["x_center_norm"], test["y_center_norm"], c="black", s=5, alpha=.5)
ax.set_title("train / test の中心点")

ax = axes[1]
# score は log-scale なので exponential しています
ax.scatter(xx.reshape(-1,), yy.reshape(-1, ), c=np.exp(zz.reshape(-1,)), s=50)
ax.set_xlim(ax.set_ylim(0, 1))
ax.set_title("KDE＠train")

fig.tight_layout()