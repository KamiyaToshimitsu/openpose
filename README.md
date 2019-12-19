# Realtime Multi-Person Pose Estimation
This is a keras version of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) project  
### discriminator3.py
敵対学習と同じ形でdiscriminatorの未学習するようにしたもの，実行するとlossがnanになる．
### discriminator2.py
generatorの出力をdatasetから読み込み学習するdiscriminator．ちゃんと学習できる．
### make_data_3.py
generatorの出力をdatasetフォルダの中に書き出すファイル．出力の取得はファイル内の関数get_heatmap_pafで行う．

## training_
### train_pose.py
もとからあったファイル．openposeのトレーニングを実行
### train_pose3.py
敵対学習を実行するファイル．ここがおかしいと思われる．

## model
### cmu_model.py
もとからあったファイル．openposeのモデルの構造などを定義しているファイル．
### cmu_model_2.py
cmu_modelをtrain_pose3で使いやすいように若干書き換えたもの．ほぼ変更していないため，多分ここは大丈夫だと思われる．
