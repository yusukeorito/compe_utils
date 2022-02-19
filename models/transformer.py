#pytorchを用いたtransformer
from pyexpat import features
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

"""
n_hidden 隠れそうのノード数
n_head ヘッドの数 8か16あたりから始めると良さそう？
num_layers Transformer layerの数

"""


n_hidden = 128
n_head = 16
num_layers = 4
head_n_hidden = 128
class Dataset(Dataset):
    def __init__(self, df, features, seq_len=15, is_train=True):

        self.bcids = df["bcid"].unique()
        self.features = features
        self.seq_len = seq_len
        self.is_train = is_train

        self.bcid_len = df["bcid"].value_counts().to_dict()
        self.features_dict = {}
        for feat in features:
            self.features_dict[feat] = df.groupby("bcid")[feat].apply(lambda x: x.values)

        if is_train:
            self.targets_dict = df.groupby("bcid")["class_id"].apply(lambda x: x.values)

    def __len__(self):
        return len(self.bcids)

    def pad_feature(self, feature):
        pad_feature = np.zeros(self.seq_len)
        pad_feature[:len(feature)] = feature
        return pad_feature

    def __getitem__(self, idx):
        bcid = self.bcids[idx]

        features = {}

        for feat in self.features:
            feat_arr = self.features_dict[feat][bcid]
            features[feat] = torch.tensor(
                self.pad_feature(feat_arr), dtype=torch.float
            )

        mask = torch.ones(self.seq_len, dtype=torch.bool)
        mask[:self.bcid_len[bcid]] = 0

        if self.is_train:
            target = torch.tensor(
                self.pad_feature(self.targets_dict[bcid]),
                dtype=torch.long
            )
            return features, mask, target
        else:
            return features, mask

class Model(nn.Module): 
    def __init__(self, features, seq_len, device, n_hidden, n_head, num_layers, head_n_hidden):
        super(Model, self).__init__()

        self.features = features
        self.feature_len = len(features)
        self.seq_len = seq_len
        self.device = device

        self.seq_linear = nn.Sequential(
            nn.Linear(self.feature_len, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(n_hidden, head_n_hidden),
            nn.LayerNorm(head_n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(head_n_hidden, 10),
        )

    def __call__(self, features, mask):

        # sequence
        seq_input = torch.cat(
            [features[f].unsqueeze(-1) for f in self.features], dim=-1
        )  # (batchsize, seq_len, feature_size)
        seq_hidden = self.seq_linear(seq_input)  # (batchsize, seq_len, n_hidden)
        seq_hidden = seq_hidden.permute(1, 0, 2)
        hidden = self.encoder(seq_hidden, src_key_padding_mask=mask)
        hidden = hidden.permute(1, 0, 2)
        pred = self.head(hidden)
        return pred




