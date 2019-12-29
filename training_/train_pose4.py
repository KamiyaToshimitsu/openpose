import math
import os
import re
import sys
import pandas
import scipy
import math
import random

from functools import partial
from tqdm import tqdm

import keras.backend as K
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.applications.vgg19 import VGG19
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from config_reader import config_reader
from keras.optimizers import Adam, Adagrad, RMSprop, SGD

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cmu_model_2 import get_training_model, get_testing_model
from training.optimizers import MultiSGD
from keras import optimizers 
from training.dataset import get_dataflow, batch_dataflow
from training.dataflow import COCODataPaths
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda, concatenate, merge, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.utils.np_utils import to_categorical


#matplotlib inline
import cv2
import matplotlib
import pylab as plt
import numpy as np
import util
import glob


batch_size = 32
#batch_size = 16
base_lr = 4e-9 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

#stepsize = 68053
#max_iter = 600000

weights_best_file_gen = "model.h5"
weights_best_file_dis ="../model_gen/model_weight.hdf5"
training_log = "training.csv"
logs_dir = "./logs"

from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def get_last_epoch():
    """
    Retrieves last epoch from log file updated during training.

    :return: epoch number
    """
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)


def restore_weights(weights_best_file, model):
    """
    Restores weights from the checkpoint file if exists or
    preloads the first layers with VGG19 weights

    :param weights_best_file:
    :return: epoch number to use to continue training. last epoch + 1 or 0
    """
    # load previous weights or vgg19 if this is the first run
    if os.path.exists(weights_best_file):
        print("Loading the best weights...")

        model.load_weights(weights_best_file)

        #return get_last_epoch() + 1
        return 0
    
    else:
        print("Loading vgg19 weights...")

        vgg_model = VGG19(include_top=False, weights='imagenet')

        for layer in model.layers:
            if layer.name in from_vgg:
                vgg_layer_name = from_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                print("Loaded VGG19 layer: " + vgg_layer_name)

        return 0


def get_lr_multipliers(model):
    """
    Setup multipliers for stageN layers (kernel and bias)

    :param model:
    :return: dictionary key: layer name , value: multiplier
    """
    lr_mult = dict()
    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

    return lr_mult


def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """
    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["weight_stage1_L1"] = _eucl_loss
    losses["weight_stage1_L2"] = _eucl_loss
    losses["weight_stage2_L1"] = _eucl_loss
    losses["weight_stage2_L2"] = _eucl_loss
    losses["weight_stage3_L1"] = _eucl_loss
    losses["weight_stage3_L2"] = _eucl_loss
    losses["weight_stage4_L1"] = _eucl_loss
    losses["weight_stage4_L2"] = _eucl_loss
    losses["weight_stage5_L1"] = _eucl_loss
    losses["weight_stage5_L2"] = _eucl_loss
    losses["weight_stage6_L1"] = _eucl_loss
    losses["weight_stage6_L2"] = _eucl_loss

    return losses


def step_decay(epoch, iterations_per_epoch):
    """
    Learning rate schedule - equivalent of caffe lr_policy =  "step"

    :param epoch:
    :param iterations_per_epoch:
    :return:
    """
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

    return lrate

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def get_max_min(picsize_bf,picsize,model):
    images_comp = glob.glob('../dataset/blender6/*')
    images_comp = random.sample(images_comp,100)
    images_real = glob.glob('../dataset/train2017/*')
    images_real = random.sample(images_real,100)
    images = images_comp+images_real

    oriImg = cv2.imread(images[0])
    oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    oriImg = oriImg.reshape(1,picsize_bf[0],picsize_bf[1],3)
    paf, pcm  = model.predict(oriImg)
    pcm = pcm.reshape(picsize[0],picsize[1],19)
    paf = paf.reshape(picsize[0],picsize[1],38)
    pcm = cv2.resize(pcm, dsize=(picsize[0],picsize[1]))
    paf = cv2.resize(paf, dsize=(picsize[0],picsize[1]))

    Max = max([np.amax(pcm),np.amax(paf)])
    Min = min([np.amin(pcm),np.amin(paf)])

    Max = max([abs(Max),abs(Min)])

    for image in tqdm(images[1:]):
        oriImg = cv2.imread(image)
        oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
        oriImg = oriImg.reshape(1,picsize_bf[0],picsize_bf[1],3)
        paf, pcm  = model.predict(oriImg)
        pcm = pcm.reshape(picsize[0],picsize[1],19)
        paf = paf.reshape(picsize[0],picsize[1],38)
        pcm = cv2.resize(pcm, dsize=(picsize[0],picsize[1]))
        paf = cv2.resize(paf, dsize=(picsize[0],picsize[1]))

        Max = max(Max,max([np.amax(pcm),np.amax(paf)]))
        Min = max(Min,min([np.amin(pcm),np.amin(paf)]))

        Max = max([abs(Max),abs(Min)])

    return Max


def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i

'''
            
def discriminator():
    inputs_face = Input(shape=(64,64,13))
    inputs_right_arm = Input(shape=(64,64,7))
    inputs_left_arm = Input(shape=(64,64,7))
    inputs_right_leg = Input(shape=(64,64,7))
    inputs_left_leg = Input(shape=(64,64,7))
    inputs_conect = Input(shape=(64,64,23))
                    

    face = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_face)
    right_arm = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_right_arm)
    left_arm = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_left_arm)
    right_leg = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_right_leg)
    left_leg = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_left_leg)
    conect = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_conect)

    face = Conv2D(64,kernel_size=5,activation='relu')(face)
    right_arm = Conv2D(64,kernel_size=5,activation='relu')(right_arm)
    left_arm = Conv2D(64,kernel_size=5,activation='relu')(left_arm)
    right_leg = Conv2D(64,kernel_size=5,activation='relu')(right_leg)
    left_leg = Conv2D(64,kernel_size=5,activation='relu')(left_leg)
    conect = Conv2D(64,kernel_size=5,activation='relu')(conect)

    face = MaxPooling2D(pool_size=(2,2))(face)
    right_arm = MaxPooling2D(pool_size=(2,2))(right_arm)
    left_arm = MaxPooling2D(pool_size=(2,2))(left_arm)
    right_leg = MaxPooling2D(pool_size=(2,2))(right_leg)
    left_leg = MaxPooling2D(pool_size=(2,2))(left_leg)
    conect = MaxPooling2D(pool_size=(2,2))(conect)

    face = Dropout(0.2)(face)
    right_arm = Dropout(0.2)(right_arm)
    left_arm = Dropout(0.2)(left_arm)
    right_leg = Dropout(0.2)(right_leg)
    left_leg = Dropout(0.2)(left_leg)
    conect = Dropout(0.2)(conect)

    face = Flatten()(face)
    right_arm = Flatten()(right_arm)
    left_arm = Flatten()(left_arm)
    right_leg = Flatten()(right_leg)
    left_leg = Flatten()(left_leg)
    conect = Flatten()(conect)

    face = Dense(256,activation='relu')(face)
    right_arm = Dense(256,activation='relu')(right_arm)
    left_arm = Dense(256,activation='relu')(left_arm)
    right_leg = Dense(256,activation='relu')(right_leg)
    left_leg = Dense(256,activation='relu')(left_leg)
    conect = Dense(256,activation='relu')(conect)

    arm = concatenate([right_arm,left_arm])
    leg = concatenate([right_leg,left_leg])

    arm = Dense(256,activation='relu')(arm)
    leg = Dense(256,activation='relu')(leg)

    merged = concatenate([face,arm,leg,conect])

    merged = Dense(256,activation='relu')(merged)

    output = Dense(1,activation='sigmoid')(merged)
    
    return Model(inputs=[inputs_face,inputs_right_arm,inputs_left_arm,inputs_right_leg,inputs_left_leg,inputs_conect],outputs=output,name='discriminator')
'''

def discriminator():
    with tf.variable_scope("discriminator"):
        inputs1 = Input(shape=(64,64,19), name='D_input_1')
        inputs2= Input(shape=(64,64,38), name='D_inputs_2')

        #inputs_pcm = BatchNormalization(axis=2)(Reshape((64,64,19))(inputs1))
        #inputs_paf = BatchNormalization(axis=2)(Reshape((64,64,38))(inputs2))

        inputs_pcm = Reshape((64,64,19))(inputs1)
        inputs_paf = Reshape((64,64,38))(inputs2)

        x = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_pcm)
        y = Conv2D(64,kernel_size=5,padding='same',activation='relu')(inputs_paf)

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
    return model



def get_mid_outputs(model_output):
    
    picsize = [64,64]
    picsize_bf = [512,512]
    param, model_params = config_reader()

    multiplier = [x * model_params['boxsize'] / picsize_bf[0] for x in param['scale_search']]

    heatmap_avg = np.zeros((picsize_bf[0], picsize_bf[1], 19))
    paf_avg = np.zeros((picsize_bf[0], picsize_bf[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(model_output, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
       
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
        #print("Input shape: " + str(input_img.shape))  

        output_blobs = model_output
        #print("Output shape (heatmap): " + str(output_blobs[1].shape))

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

        #pcm_cmp = pcm_cmp.reshape(picsize[0],picsize[1],19).astype("float32")
        #paf_cmp = paf_cmp.reshape(picsize[0],picsize[1],38).astype("float32")
        
    return heatmap_avg,paf_avg

def make_dis_inputs(heatmaps,pafs):
    images1 = heatmaps
    images2 = pafs

    images_face = tf.concat((images1[:,:,[0,14,15,16,17]],images2[:,:,[30,31,32,33,34,35,36,37]]),axis=2)
    images_right_arm = tf.concat((images1[:,:,[2,3,4]],images2[:,:,[14,15,16,17]]),axis=2)
    images_left_arm = tf.concat((images1[:,:,[5,6,7]],images2[:,:,[22,23,24,25]]),axis=2)
    images_right_leg = tf.concat((images1[:,:,[8,9,10]],images2[:,:,[2,3,4,5]]),axis=2)
    images_left_leg = tf.concat((images1[:,:,[11,12,13]],images2[:,:,[8,9,10,11]]),axis=2)
    images_conect = tf.concat((images1[:,:,[0,1,2,5,8,11,16,17,18]],images2[:,:,[0,1,6,7,12,13,18,19,20,21,26,27,28,29]]),axis=2)
    
    return [inputs_face,inputs_right_arm,inputs_left_arm,inputs_right_leg,inputs_left_leg,inputs_conect]    



def generator():

    # get the model
    with tf.variable_scope("generator"):
        inputs_paf, inputs_pcm, img_input, gen = get_testing_model()
    return gen
    

#モデルを学習するかどうかを制御
def set_trainable(model,trainable=False):
    model.trainable = trainable
    try:
        layers = model.layers
    except:
        return
    for layer in layers:
        set_trainable(layer, trainable)

#def gen_binary_crossentropy(y_true, y_pred):
    #return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def GAN(gen = generator(),dis = discriminator()):
    with tf.variable_scope("GAN"):
        inputs = Input(shape = (None, None, 3), name = 'GAN_input')
        x,y = gen(inputs)
        output = dis([y,x])
        model = Model(inputs=inputs,outputs=output)
    return model





if __name__ == '__main__':

    '''
    # restore weights

    last_epoch = restore_weights(weights_best_file, model)

    # prepare generators

    curr_dir = os.path.dirname(__file__)
    annot_path_train = os.path.join(curr_dir, '../dataset/annotations/blender_real.json')
    img_dir_train = os.path.abspath(os.path.join(curr_dir, '../dataset/blender_real/'))
    annot_path_val = os.path.join(curr_dir, '../dataset/annotations/blender_real_val.json')
    img_dir_val = os.path.abspath(os.path.join(curr_dir, '../dataset/blender_real_val/'))

    # get dataflow of samples from training set and validation set (we use validation set for training as well)

    coco_data_train = COCODataPaths(
        annot_path=annot_path_train,
        img_dir=img_dir_train
    )
    coco_data_val = COCODataPaths(
        annot_path=annot_path_val,
        img_dir=img_dir_val
    )
    df = get_dataflow([coco_data_train, coco_data_val])
    train_samples = df.size()

    # get generator of batches

    batch_df = batch_dataflow(df, batch_size)
    train_gen = gen(batch_df)

    # setup lr multipliers for conv layers

    lr_multipliers = get_lr_multipliers(model)

    # configure callbacks

    iterations_per_epoch = train_samples // batch_size
    _step_decay = partial(step_decay,
                          iterations_per_epoch=iterations_per_epoch
                          )
    lrate = LearningRateScheduler(_step_decay)
    checkpoint = ModelCheckpoint(weights_best_file, monitor='loss',
                                 verbose=0, save_best_only=False,
                                 save_weights_only=True, mode='min', period=1)
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                     write_images=False)
    
    #set early stopping
    es_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    
    callbacks_list = [lrate, checkpoint, csv_logger, tb, es_cb]

    # sgd optimizer with lr multipliers

    multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
                        nesterov=False, lr_mult=lr_multipliers)

    #multisgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    '''

    # start training
    # GPU configulations
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    # random seeds
    np.random.seed(1)
    tf.set_random_seed(1)

    # parameters
    max_epoch = 101
    batch_size = 10
    picsize_bf = [512,512]
    picsize = [64,64]

    # model definitions
    print('Model definitions...')
    gen = generator()
    dis = discriminator()
    gen.load_weights(weights_best_file_gen)
    #dis.load_weights(weights_best_file_dis)
    lr_multipliers = get_lr_multipliers(gen)
    com = GAN(gen=gen,dis=dis)
    gen_opt = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_multipliers)
    dis_opt = Adam(lr = 0.001)

    # generator compile 
    gen.compile(loss = 'binary_crossentropy',optimizer=gen_opt) 
    set_trainable(dis,False)
    set_trainable(gen,True)
    com.compile(loss = 'binary_crossentropy', optimizer=dis_opt, metrics = ["accuracy"])

    #discriminator compile
    set_trainable(dis,True)
    set_trainable(gen,False)
    dis.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics = ["accuracy"])


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
    y_true2 = y_true + y_true 

    print(len(X_train))
    print(len(y_train))

    train_zip = list(zip(X_train,y_train))

    Max = get_max_min(picsize_bf,picsize,gen)
    print(Max)

    #plot_model(gen_train_stage.layers[1], to_file='model.png', show_shapes=True, show_layer_names=True)

    logging = TensorBoard(log_dir = "log_3/")
    logging.set_model(gen)
    train_names = ['train_loss', 'train_mae']


    # for each epoch
    history = dict()
    print('Training...')
    for epoch in tqdm(range(max_epoch)):
        # random permutation
        #ns = np.random.permutation(n_samples_train)
        random.shuffle(X_train_cmp)
        random.shuffle(train_zip)
        X_train,y_train = zip(*train_zip)
        # for each batch
        gen_loss = 0
        gen_acc = 0
        dis_loss = 0
        dis_acc = 0
        num_batches = int(math.floor(n_samples_train / batch_size))

        # discriminator training stage
        for batch in tqdm(range(num_batches)):
            # batch size
            batch_start = batch * batch_size
            batch_end = min((batch+1)*batch_size, n_samples_train)
            now_batch_size = batch_end - batch_start

            # batch of real images
            X_train_batch = cv2.imread(X_train[batch_start])
            X_train_batch = cv2.resize(X_train_batch, dsize=(picsize_bf[0],picsize_bf[1]))
            X_train_batch = X_train_batch.reshape(1,X_train_batch.shape[0],X_train_batch.shape[1],X_train_batch.shape[2])
            for i_x in X_train[batch_start+1:batch_end]:
                i_x = cv2.imread(i_x)
                i_x = cv2.resize(i_x, dsize=(picsize_bf[0],picsize_bf[1]))
                i_x = i_x.reshape(1,i_x.shape[0],i_x.shape[1],i_x.shape[2])
                X_train_batch = np.vstack((X_train_batch, i_x))
            #X_train_batch = X_train_real[batch_start:batch_end] 

            X_paf,X_pcm = gen.predict(X_train_batch)
            X_paf_batch,X_pcm_batch = X_paf/Max,X_pcm/Max           

            # batch of discriminator labels
            Y_train_batch = np.array(y_train[batch_start:batch_end])            
            #print(np.amax(X_train_batch))
                       
            dis_loss_now, dis_acc_now = dis.train_on_batch([X_pcm_batch,X_paf_batch], Y_train_batch)
            dis_loss += dis_loss_now
            dis_acc  += dis_acc_now

            print('\r dis_loss = {0:.3f}, dis_acc = {1:.3f}'.format(dis_loss_now,dis_acc_now), end='')
            sys.stdout.flush()
        # end for batch in tqdm(range(num_batches))

        # generator training stage
        for batch in tqdm(range(num_batches)):
            # batch size
            batch_start = batch * batch_size
            batch_end = min((batch+1)*batch_size, n_samples_train)
            now_batch_size = batch_end - batch_start

            # batch of cmp image
            Z_train_batch = cv2.imread(X_train_cmp[batch_start])
            Z_train_batch = cv2.resize(Z_train_batch, dsize=(picsize_bf[0],picsize_bf[1]))
            Z_train_batch = Z_train_batch.reshape(1,Z_train_batch.shape[0],Z_train_batch.shape[1],Z_train_batch.shape[2])
            for i_z in X_train_cmp[batch_start+1:batch_end]:
                i_z = cv2.imread(i_z)
                i_z = cv2.resize(i_z, dsize=(picsize_bf[0],picsize_bf[1]))
                i_z = i_z.reshape(1,i_z.shape[0],i_z.shape[1],i_z.shape[2])
                Z_train_batch = np.vstack((Z_train_batch, i_z))
            #Z_train_batch = X_train_cmp[batch_start:batch_end]

            # batch of generator labels
            Y_fake_batch = np.array(y_true2[batch_start:batch_end])
          
            gen_loss_now, gen_acc_now = com.train_on_batch(Z_train_batch, Y_fake_batch)
            gen_loss += gen_loss_now
            gen_acc  += gen_acc_now

            print('\r gen_loss = {0:.3f}, gen_acc = {1:.3f}'.format(gen_loss_now,gen_acc_now), end='')
            sys.stdout.flush()




        # loss and accuracy
        gen_loss /= num_batches
        dis_loss /= num_batches
        print('Generator training loss:         {0}'.format(gen_loss))
        print('Discriminator training loss:     {0}'.format(dis_loss))
        gen_acc /= num_batches
        dis_acc /= num_batches
        print('Generator training accuracy:     {0}'.format(gen_acc))
        print('Discriminator training accuracy: {0}'.format(dis_acc))

        # training history
        history[epoch] = {
            'gen_train_loss': gen_loss, 'dis_train_loss': dis_loss, 
            'gen_train_acc' : gen_acc,  'dis_train_acc' : dis_acc,  
        }
        logs = [dis_loss,dis_acc]
        write_log(logging, train_names, logs, batch)
         # end for epoch in tqdm(range(max_epoch))

    # save the results
    ## model definitions
    with open('modeldefs.json', 'w') as fout:
        model_json_str = gen_train_stage.to_json(indent=4)
        fout.write(model_json_str)
    ## model weights
    gen.save_weights('modelweights.hdf5')
    dis.save_weights('modelweights_dis.hdf5')
    ## training history
    json.dump(history, open('training_history.json', 'w'))
