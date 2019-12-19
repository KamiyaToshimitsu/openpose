import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import pathlib
import glob
import random
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import concatenate, merge, Input
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10

class ImageDataGenerator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.images1 = np.empty((1,64,64,19))
        self.images2 = np.empty((1,64,64,38))
        self.labels = []

    def flow_from_directory(self, directory_pcm ,directory_paf, classes, batch_size):
        # LabelEncode(classをint型に変換)するためのdict
        classes = {v: i for i, v in enumerate(sorted(classes))}
        while True:
            # ディレクトリから画像のパスを取り出す
            for path_pcm in directory_pcm:
                path_pcm = pathlib.Path(path_pcm)
                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納
                self.images1 = np.vstack((self.images1, np.load(path_pcm).reshape(1,64,64,19)))
                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納
                self.images2 = np.vstack((self.images2, np.load(os.path.join(str(directory_paf),os.path.basename(str(path_pcm)))).reshape(1,64,64,38)))
                
                # ファイル名からラベルを取り出し、配列(self.labels)に格納
                _, y = path_pcm.stem.split('_')
                if y == "1":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納
                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする　　　
                if len(self.images1) == batch_size+1:
                    inputs1 = np.asarray(self.images1)
                    inputs2 = np.asarray(self.images2)
                    inputs1 = np.delete(inputs1,0,0)
                    inputs2 = np.delete(inputs2,0,0)
                    targets = np.asarray(self.labels)
                    inputs1 = inputs1.reshape(batch_size,64,64,19).astype("float32")
                    inputs2 = inputs2.reshape(batch_size,64,64,38).astype("float32")
                    targets = targets.astype("float32")
                    self.reset()
                    yield [inputs1, inputs2], targets

class ImageDataGenerator_val(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.images1 = np.empty((1,64,64,19))
        self.images2 = np.empty((1,64,64,38))
        self.labels = []

    def flow_from_directory(self, directory_pcm ,directory_paf, classes, batch_size):
        # LabelEncode(classをint型に変換)するためのdict
        classes = {v: i for i, v in enumerate(sorted(classes))}
        while True:
            # ディレクトリから画像のパスを取り出す
            for path_pcm in directory_pcm:
                path_pcm = pathlib.Path(path_pcm)
                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納
                self.images1 = np.vstack((self.images1, np.load(path_pcm).reshape(1,64,64,19)))
                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納
                self.images2 = np.vstack((self.images2, np.load(os.path.join(str(directory_paf),os.path.basename(str(path_pcm)))).reshape(1,64,64,38)))
                
                # ファイル名からラベルを取り出し、配列(self.labels)に格納
                _, y = path_pcm.stem.split('_')
                if y == "1":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納
                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする　　　
                if len(self.images1) == batch_size+1:
                    inputs1 = np.asarray(self.images1)
                    inputs2 = np.asarray(self.images2)
                    inputs1 = np.delete(inputs1,0,0)
                    inputs2 = np.delete(inputs2,0,0)
                    targets = np.asarray(self.labels)
                    inputs1 = inputs1.reshape(batch_size,64,64,19).astype("float32")
                    inputs2 = inputs2.reshape(batch_size,64,64,38).astype("float32")
                    targets = targets.astype("float32")
                    self.reset()
                    yield [inputs1, inputs2], targets


batch_size=32
epochs=128

inputs1 = Input(shape=(64,64,19))
inputs2= Input(shape=(64,64,38))

x = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs1)
y = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs2)

x = Conv2D(64,kernel_size=5,activation='relu')(x)
y = Conv2D(64,kernel_size=5,activation='relu')(y)

x = MaxPooling2D(pool_size=(2,2))(x)
y = MaxPooling2D(pool_size=(2,2))(y)

x = Dropout(0.2)(x)
y = Dropout(0.2)(y)

x = Flatten()(x)
y = Flatten()(y)

x = Dense(256,activation='relu')(x)
y = Dense(256,activation='relu')(y)

merged = concatenate([x,y])

merged = Dense(64,activation='relu')(merged)

output = Dense(1,activation='sigmoid')(merged)

model = Model(inputs=[inputs1,inputs2],outputs=output)

optimizer = Adam(lr = 0.001)
model.compile(
    optimizer = optimizer,
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
)

weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

logging = TensorBoard(log_dir = "log/")

pcm_train = glob.glob('dataset/pcm_train/*')
paf_train = pathlib.Path('dataset/paf_train/')
random.shuffle(pcm_train)
pcm_train = pcm_train[:int(len(pcm_train)*0.7)]
pcm_val = pcm_train[int(len(pcm_train)*0.7):]
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator_val()
classes = ['0', '1']

model.fit_generator(
    generator=train_datagen.flow_from_directory(pcm_train, paf_train, classes,batch_size),
    steps_per_epoch=int(np.ceil(len(list(pcm_train))/ batch_size)),
    validation_data = val_datagen.flow_from_directory(pcm_val, paf_train, classes, batch_size),
    validation_steps = int(np.ceil(len(list(pcm_val))/ batch_size)),
    epochs=epochs,
    callbacks = [logging,early_stopping],
    verbose=1)

model_dir = './model_gen/'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

model.save(model_dir + 'model.hdf5')

# optimizerのない軽量モデルを保存（学習や評価は不可だが、予測は可能）
model.save(model_dir + 'model-opt.hdf5', include_optimizer = False)

# ベストの重みのみ保存
model.save_weights(model_dir + 'model_weight.hdf5')

#accuracyの出力
pcm_test = pathlib.Path('dataset/pcm_test/*')
paf_test = pathlib.Path('dataset/paf_test/')
random.shuffle(pcm_train)
train_datagen = ImageDataGenerator()
classes = ['0', '1']

score = model.evaluate_generator(generator=train_datagen.flow_from_directory(pcm_test, paf_test, classes,batch_size), steps=int(np.ceil(len(list(pcm_test))/ batch_size)), verbose=1)
print("evaluate loss: {0[0]}".format(score))
print("evaluate acc: {0[1]}".format(score))
