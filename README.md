# コンペ用のコード 
### Utils:実験管理やディレクトリ生成などの汎用的なコード。notebook用。
make_folder.py:実験管理用のフォルダ作成
logger.py:モデルや特徴量の読み込みと記録
time.py:Timer


### feature
basic_feature.py:数値特徴量,count_encoding, onehot_encoding, target_encoding,  
statistic_feature.py:各種統計量をとる特徴量<br>
create_feature.py:指定blockの特徴量の作成

### model:モデルのラッパー。 train_cv内で使用できるような形にアレンジしている。

