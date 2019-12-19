# Realtime Multi-Person Pose Estimation
This is a keras version of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) project  

### discriminator2.py
generatorの出力をdatasetから読み込み学習するdiscriminator．ちゃんと学習できる．
### discriminator3.py
敵対学習と同じ形でdiscriminatorの未学習するようにしたもの，実行するとlossがnanになる．
discriminator2との違いは233行目以降，バグの原因はそれ以降か，175行目～194行目でgeneratorとdiscriminatorを接続している部分だと思われる．
### make_data_3.py
generatorの出力をdatasetフォルダの中に書き出すファイル．出力の取得はファイル内の関数get_heatmap_pafで行う．

## training_
### train_pose.py
もとからあったファイル．openposeのトレーニングを実行
### train_pose3.py
敵対学習を実行するファイル．ここがおかしいと思われる．
516行目のパスはdataset内の合成画像が格納されているフォルダ，519行目のパスはdataset内の実画像の格納されているフォルダを示す．
画像は入力前に255で除算して正規化している．

## model
### cmu_model.py
もとからあったファイル．openposeのモデルの構造などを定義しているファイル．
### cmu_model_2.py
cmu_modelをtrain_pose3で使いやすいように若干書き換えたもの．ほぼ変更していないため，多分ここは大丈夫だと思われる．
