import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import cv2
import os
import sys
import pathlib
import glob
import random
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import concatenate, merge, Input, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.cmu_model_2 import get_training_model, get_testing_model

from functools import partial
from tqdm import tqdm

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




batch_size=10
epochs=128
picsize_bf = [512,512]
weights_best_file_gen = "model.h5"

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

# designate trainable layers
def set_trainable(model, trainable=False):
    model.trainable = trainable
    try:
        layers = model.layers
    except:
        return
    for layer in layers:
        set_trainable(layer, trainable)

def generator():

    # get the model
    return get_testing_model()

def discriminator():
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

    return Model(inputs=[inputs1,inputs2],outputs=output)



inputs = Input(shape=(None,None,3))

gen = generator()
gen.load_weights(weights_best_file_gen)

inputs_paf,inputs_pcm = gen(inputs)
#inputs_pcm = BatchNormalization(axis=2)(Reshape((64,64,19))(inputs_pcm))
#inputs_paf = BatchNormalization(axis=2)(Reshape((64,64,38))(inputs_paf))
inputs_pcm = BatchNormalization(axis=3)(inputs_pcm)
inputs_paf = BatchNormalization(axis=3)(inputs_paf)
gen_outputs = [inputs_pcm,inputs_paf]


dis = discriminator()

dis_outputs = dis(gen_outputs)

set_trainable(gen, False)
set_trainable(dis, True)
model = Model(inputs,dis_outputs)



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

logging = TensorBoard(log_dir = "log_2/")
logging.set_model(model)
train_names = ['train_loss', 'train_mae']

# load the dataset
print('Loading the dataset...')
X_train_cmp = glob.glob('../dataset/blender6/*')
X_train_cmp = random.sample(X_train_cmp,3000)
y_fake = list(np.zeros(3000))

X_train_real = glob.glob('../dataset/train2017/*') 
X_train_real = random.sample(X_train_real,3000)
y_true = list(np.ones(3000))

n_samples_train = len(X_train_cmp)

X_train = X_train_cmp + X_train_real
y_train = y_fake + y_true 

print(len(X_train))
print(len(y_train))

train_zip = list(zip(X_train,y_train))

for epoch in tqdm(range(epochs)):

    random.shuffle(train_zip)
    X_train,y_train = zip(*train_zip)


    loss = 0
    acc = 0

    num_batches = int(math.floor(n_samples_train / batch_size))
    for batch in tqdm(range(num_batches)):
        # batch size
        batch_start = batch * batch_size
        batch_end = min((batch+1)*batch_size, n_samples_train)
        now_batch_size = batch_end - batch_start
        # batch of real images
        X_train_batch = cv2.imread(X_train[batch_start])
        X_train_batch = cv2.resize(X_train_batch, dsize=(picsize_bf[0],picsize_bf[1]))
        X_train_batch = X_train_batch/255
        X_train_batch = X_train_batch.reshape(1,X_train_batch.shape[0],X_train_batch.shape[1],X_train_batch.shape[2])
        for i_x in X_train[batch_start+1:batch_end]:
            i_x = cv2.imread(i_x)
            i_x = cv2.resize(i_x, dsize=(picsize_bf[0],picsize_bf[1]))
            i_x = i_x/255
            i_x = i_x.reshape(1,i_x.shape[0],i_x.shape[1],i_x.shape[2])
            X_train_batch = np.vstack((X_train_batch, i_x))
        # batch of discriminator labels
        Y_train_batch = np.array(y_train[batch_start:batch_end])

        loss_now, acc_now = model.train_on_batch(X_train_batch,Y_train_batch)
        loss += loss_now
        acc += acc_now
        print("\r loss = {0:.3f}, acc = {1:.3f}".format(loss_now,acc_now),end='')
        sys.stdout.flush()
    loss /= num_batches
    print('training loss:         {0}'.format(loss))
    acc /= num_batches
    print('training accuracy:     {0}'.format(acc))
    logs = [loss,acc]
    write_log(logging, train_names, logs, batch)





model_dir = './model_gen/'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

model.save(model_dir + 'model.hdf5')

# optimizerのない軽量モデルを保存（学習や評価は不可だが、予測は可能）
model.save(model_dir + 'model-opt.hdf5', include_optimizer = False)

# ベストの重みのみ保存
model.save_weights(model_dir + 'model_weight.hdf5')

'''
#accuracyの出力
pcm_test = glob.glob('dataset/pcm_test/*')
paf_test = pathlib.Path('dataset/paf_test/')
random.shuffle(pcm_test)
train_datagen = ImageDataGenerator()
classes = ['0', '1']

score = model.evaluate_generator(generator=train_datagen.flow_from_directory(pcm_test, paf_test, classes, batch_size), steps=int(np.ceil(len(list(pcm_test))/ batch_size)), verbose=1)
print("evaluate loss: {0[0]}".format(score))
print("evaluate acc: {0[1]}".format(score))
'''