import os
import random
import numpy as np
import torch

def get_logger(filename='log'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__) #loggerインスタンスの生成
    logger.setLevel(INFO)
    handler1 = StreamHandler()   #ターミナルに出力するhandler
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log") #logファイルとして出力するhandler
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()

#seed固定関数
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed) #pythonがハッシュを作成するときのシード
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)


    