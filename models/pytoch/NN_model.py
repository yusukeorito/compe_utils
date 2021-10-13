#PytorchによるMLPの実装(テーブルデータ向け)

import os
import sys
import pandas as pd
import numpy as np
from models.pytoch.setting import get_logger, seed_everything

from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

sys.path.append('.')
import setting

#Loading Data
train = pd.read_csv('')
test = pd.read_csv('')
sub = pd.read_csv('')

#CV split
folds = train.copy()
Fold = KFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_idx, valid_idx) in enumerate(Fold.split(folds, folds[target_cols])):
    folds.loc[valid_idx, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)


#Dataset
class TrainDataset(Dataset):
    def __init__(self, df, num_features, cat_features, labels):
        self.cont_values = df[num_features].values #numpy array
        self.cat_values  = df[cat_features].values
        self.labels = labels

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, index):
        cont_x = torch.FloatTensor(self.cont_values[index]) #torch tensor型に変更
        cat_x  = torch.LongTensor(self.cat_values[index])
        label = torch.tensor(self.labels[index]).float()
        return cont_x, cat_x, label

    
class TestDataset(Dataset):
    def __init__(self,df, num_features, cat_features) -> None:
        self.cont_values = df[num_features].values
        self.cat_values  = df[cat_features].values
    
    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, index):
        cont_x = torch.FloatTensor(self.cont_values[index])
        cat_x  = torch.LongTensor(self.cat_values[index])
        return cont_x, cat_x


#Model
class Config:
    hidden_size = 512
    drop_out = 0.5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    lr = 1e-2
    batch_size = 30
    epochs = 20
    n_folds = 5
    num_features = num_features
    cat_features = cat_features
    target_cols = target_cols

class TabularNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(cfg.num_features), cfg.hidden_size),
            nn.BatchNorm1d(cfg.hidden_size),
            nn.Dropout(cfg.drop_out),
            nn.PReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.BatchNorm1d(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.PReLU(),
            nn.Linear(cfg.hidden_size, len(cfg.target_cols)),
        )

    def forward(self, cont_x, cat_x):
        x = self.mlp(cont_x)
        return x

#Training 
def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    losses = AverageMeter()

    model.train()

    for step , (cont_x, cat_x, y) in enumerate(train_loader):
        cont_x, cat_x, y = cont_x.to(device), cat_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        pred = model(cont_x, cat_x)
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        
        loss.backward()

        grad_norn = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        
    return losses.avg #1epochごとにiterationのlossの平均値を返す


def valid_fn(valid_loader, model, device):
    
    losses = AverageMeter()

    model.eval()
    val_preds = []

    for step, (cont_x, cat_x, y) in enumerate(valid_loader):

        cont_x, cat_x, y = cont_x.to(device), cat_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cat_x)
        
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        val_preds.append(pred.sigmoid().detach().cpu().numpy())

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        
    val_preds = np.concatenate(val_preds)

    return losses.avg, val_preds


def inference_fn(test_loader, model, device):
    model.eval()
    preds = []

    for step, (cont_x, cat_x) in enumerate(test_loader):

        cont_x, cat_x = cont_x.to(device), cat_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cat_x)

        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds




class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run(cfg, train, test, folds, num_features, cat_features, target, device, fold_num=0, seed=42):
    #set seed
    logger = get_logger()
    logger.info(f"Set seed {seed}") 
    seed_everything(seed=seed)

    #loader
    train_idx = folds[folds['fold'] != fold_num].index
    valid_idx = folds[folds['fold'] == fold_num].index
    train_folds = train.iloc[train_idx].reset_index(drop=True)
    valid_folds = train.iloc[valid_idx].reset_index(drop=True)
    train_target = target[train_idx]
    valid_target = target[valid_idx]
    train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    
    model = TabularNN(cfg)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimzier=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=cfg.epochs, steps_per_epoch=len(train_loader))

    log_df = pd.DataFrame(columns=[['EPOCH']+['TRAIN_LOSS']+['VALID_LOSS']])

    best_loss = np.inf

    for epoch in range(cfg.epochs):
        train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)
        valid_loss, valid_preds = valid_fn(valid_loader, model, device)
        log_row = {'EPOCH':epoch,'TRAIN_LOSS':train_loss, 'VALID_LOSS':valid_loss}
        log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)

        if valid_loss < best_loss:
            logger.info(f'epoch {epoch} save best model ... {valid_loss}')
            best_loss = valid_loss
            oof = np.zeros((len(train), len(cfg.target_cols)))
            oof[valid_idx] = valid_preds
            torch.save(model.state_dict(), f"fold{fold_num}_seed_{seed}.pth") #学習したモデルのパラメータを保存

    
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = TabularNN(cfg)
    model.load_state_dict(torch.load(f"fold{fold_num}_seed{seed}.pth"))
    predictions = inference_fn(test_loader, model, device)

    torch.cuda.empty_cache() #del

    return oof, predictions


def run_kfold(cfg, train, test, folds, num_features, cat_features, target, device, n_fold=Config.n_folds, seed=42):
    oof = np.zeros(len(train), len(cfg.target_cols))
    predictions = np.zeros((len(test), len(cfg.target_cols)))

    for _fold in range(n_fold):
        logger.info(f"Fold {}".format(_fold))
        
        
        


