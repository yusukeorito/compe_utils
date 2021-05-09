#実験管理や再現性を確保するためのフォルダ生成プログラム(notebook用)

import requests
import os

def get_exp_name():
    return requests.get('http://172.28.0.2:9000/api/sessions').json()[0]['name'][:-6]

#各種フォルダ
COLAB = "/content/drive/MyDrive/"
OUTPUT = "/content/drive/output"
INPUT = "/content/drive/input"
SUBMISSION = os.path.join(COLAB, "submission")
EXP_NAME = get_exp_name() #file名を取得
EXP = os.path.join(OUTPUT, EXP_NAME)
PREDS = os.path.join(EXP, "preds")
TRAINED = os.path.join(EXP, "trainec")
FEATURES = os.path.join(EXP, "feature")
REPORTS = os.path.join(EXP, "reports")

dirs = [
    INPUT,
    OUTPUT,
    SUBMISSION,
    EXP,
    PREDS,
    TRAINED,
    FEATURES,
    REPORTS
]

for v in dirs:
    if not os.path.isdir(v):
        print(f"making {v}")
        os.makedirs(v)
