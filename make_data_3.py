import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from config_reader import config_reader
from tqdm import tqdm
import scipy
import math
import random

#matplotlib inline
import cv2
import matplotlib
import pylab as plt
import numpy as np
import util
import glob

#Helper functions to create a model

def relu(x): 
    return Activation('relu')(x)

def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x):
     
    # Block 1
    x = conv(x, 64, 3, "conv1_1")
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1")
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    
    # Block 3
    x = conv(x, 256, 3, "conv3_1")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_2")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_3")
    x = relu(x)    
    x = conv(x, 256, 3, "conv3_4")
    x = relu(x)    
    x = pooling(x, 2, 2, "pool3_1")
    
    # Block 4
    x = conv(x, 512, 3, "conv4_1")
    x = relu(x)    
    x = conv(x, 512, 3, "conv4_2")
    x = relu(x)    
    
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM")
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = relu(x)
    
    return x

def stage1_block(x, num_p, branch):
    
    # Block 1        
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)
    
    return x

def stageT_block(x, num_p, stage, branch):
        
    # Block 1        
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))
    
    return x

#Create keras model and load weights

#weights_path = "model/keras/model.h5" # orginal weights converted from caffe
#weights_path = "training/weights.best.h5" # weights tarined from scratch 
#weights_path = "model_low_learning_rate/model.h5"
#weights_path = "model_blender2/weights.best.h5"
weights_path = "model_4e-9_cam+/model.h5"

input_shape = (None,None,3)

img_input = Input(shape=input_shape)

stages = 6
np_branch1 = 38
np_branch2 = 19

img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

# VGG
stage0_out = vgg_block(img_normalized)

# stage 1
stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

# stage t >= 2
for sn in range(2, stages + 1):
    stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
    stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
    if (sn < stages):
        x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
model.load_weights(weights_path)

def get_heatmap_paf(oriImg,picsize_bf=[512,512],picsize=[64,64]):
    oriImg = oriImg.reshape(1,picsize_bf[0],picsize_bf[1],3)
    output_blobs = model.predict(oriImg)
    output_blobs[0] = output_blobs[0].reshape(picsize[0],picsize[1],38)
    output_blobs[1] = output_blobs[1].reshape(picsize[0],picsize[1],19)
    '''
    param, model_params = config_reader()

    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        
       
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
        #print("Input shape: " + str(input_img.shape))  

        output_blobs = model.predict(input_img)
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
    '''  
    return output_blobs[1],output_blobs[0]

def preprocess(img):
    h, w, c = img.shape
    longest_edge = max(h, w)
    top = 0
    bottom = 0
    left = 0
    right = 0
    if h < longest_edge:
        diff_h = longest_edge - h
        top = diff_h // 2
        bottom = diff_h - top
    elif w < longest_edge:
        diff_w = longest_edge - w
        left = diff_w // 2
        right = diff_w - left
    else:
        pass
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

i = 0

def get_max_min(picsize_bf=[512,512],picsize=[64,64]):
    images_comp = glob.glob('dataset/blender6/*')
    images_comp = random.sample(images_comp,100)
    images_real = glob.glob('dataset/train2017/*')
    images_real = random.sample(images_real,100)
    images = images_comp+images_real

    oriImg = cv2.imread(images[0])
    oriImg = preprocess(oriImg)
    oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    pcm , paf  = get_heatmap_paf(oriImg)
    print(pcm.shape)
    pcm = cv2.resize(pcm, dsize=(picsize[0],picsize[1]))
    paf = cv2.resize(paf, dsize=(picsize[0],picsize[1]))

    Max = max([np.amax(pcm),np.amax(paf)])
    Min = min([np.amin(pcm),np.amin(paf)])

    Max = max([abs(Max),abs(Min)])

    for image in tqdm(images[1:]):
    
        oriImg = cv2.imread(image)
        oriImg = preprocess(oriImg)
        oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    
        pcm,paf = get_heatmap_paf(oriImg)

        pcm = cv2.resize(pcm, dsize=(picsize[0],picsize[1]))
        paf = cv2.resize(paf, dsize=(picsize[0],picsize[1]))

        Max = max(Max,max([np.amax(pcm),np.amax(paf)]))
        Min = max(Min,min([np.amin(pcm),np.amin(paf)]))

        Max = max([abs(Max),abs(Min)])

    return Max
    
        
    


## 合成画像取り込み

train_y_cmp = []

#discriminatorに入力する画像サイズを指定
picsize_bf = [512,512]
picsize = [64,64]

#正規化用にmaxを取得
Max = get_max_min(picsize_bf,picsize)

#指定ディレクトリの画像を取り込み
images = glob.glob('dataset/blender6/*')
images = random.sample(images,3000)
images_test = random.sample(images,500)
images_train = images

for im in images_test:
    images_train.remove(im)
    
#train画像取り込み
oriImg = cv2.imread(images_train[0])
oriImg = preprocess(oriImg)
oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
train_pcm_cmp,train_paf_cmp = get_heatmap_paf(oriImg)

pcm_cmp = cv2.resize(train_pcm_cmp, dsize=(picsize[0],picsize[1]))
paf_cmp = cv2.resize(train_paf_cmp, dsize=(picsize[0],picsize[1]))

pcm_cmp = pcm_cmp.reshape(1,picsize[0],picsize[1],19).astype("float32")
paf_cmp = paf_cmp.reshape(1,picsize[0],picsize[1],38).astype("float32")

pcm_cmp=pcm_cmp/Max
paf_cmp=paf_cmp/Max
    
np.save('dataset/pcm_train/{}_1.npy'.format(i),pcm_cmp)
np.save('dataset/paf_train/{}_1.npy'.format(i),paf_cmp)
#np.save('dataset/y/train_y_cmp_{}.npy'.format(i),y_cmp)

i += 1


for image in tqdm(images_train[1:]):
    
    oriImg = cv2.imread(image)
    oriImg = preprocess(oriImg)
    oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    
    pcm_cmp,paf_cmp = get_heatmap_paf(oriImg)

    pcm_cmp = cv2.resize(pcm_cmp, dsize=(picsize[0],picsize[1]))
    paf_cmp = cv2.resize(paf_cmp, dsize=(picsize[0],picsize[1]))
    
    pcm_cmp = pcm_cmp.reshape(1,picsize[0],picsize[1],19).astype("float32")
    paf_cmp = paf_cmp.reshape(1,picsize[0],picsize[1],38).astype("float32")
    
    pcm_cmp=pcm_cmp/Max
    paf_cmp=paf_cmp/Max
    
    np.save('dataset/pcm_train/{}_1.npy'.format(i),pcm_cmp)
    np.save('dataset/paf_train/{}_1.npy'.format(i),paf_cmp)
    #np.save('dataset/y/train_y_cmp_{}.npy'.format(i),y_cmp)
    
    i += 1
    #train_pcm_cmp = np.vstack((train_pcm_cmp,pcm_cmp))
    #train_paf_cmp = np.vstack((train_paf_cmp,paf_cmp))

"""
for y in range(len(train_pcm_cmp)):
    
    train_y_cmp.append(1)

train_y_cmp = np.array(train_y_cmp)
"""
#test画像取り込み
oriImg = cv2.imread(images_test[0])
oriImg = preprocess(oriImg)
oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
train_pcm_cmp,train_paf_cmp = get_heatmap_paf(oriImg)

pcm_cmp = cv2.resize(train_pcm_cmp, dsize=(picsize[0],picsize[1]))
paf_cmp = cv2.resize(train_paf_cmp, dsize=(picsize[0],picsize[1]))

pcm_cmp = pcm_cmp.reshape(1,picsize[0],picsize[1],19).astype("float32")
paf_cmp = paf_cmp.reshape(1,picsize[0],picsize[1],38).astype("float32")

pcm_cmp=pcm_cmp/Max
paf_cmp=paf_cmp/Max
    
np.save('dataset/pcm_test/{}_1.npy'.format(i),pcm_cmp)
np.save('dataset/paf_test/{}_1.npy'.format(i),paf_cmp)
#np.save('dataset/y/train_y_cmp_{}.npy'.format(i),y_cmp)

i += 1


for image in tqdm(images_test[1:]):
    
    oriImg = cv2.imread(image)
    oriImg = preprocess(oriImg)
    oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    
    pcm_cmp,paf_cmp = get_heatmap_paf(oriImg)

    pcm_cmp = cv2.resize(pcm_cmp, dsize=(picsize[0],picsize[1]))
    paf_cmp = cv2.resize(paf_cmp, dsize=(picsize[0],picsize[1]))
    
    pcm_cmp = pcm_cmp.reshape(1,picsize[0],picsize[1],19).astype("float32")
    paf_cmp = paf_cmp.reshape(1,picsize[0],picsize[1],38).astype("float32")
    
    pcm_cmp=pcm_cmp/Max
    paf_cmp=paf_cmp/Max
    
    np.save('dataset/pcm_test/{}_1.npy'.format(i),pcm_cmp)
    np.save('dataset/paf_test/{}_1.npy'.format(i),paf_cmp)
    
    i += 1

    
## 実画像取り込み

train_y_real = []

#discriminatorに入力する画像サイズを指定
#picsize = [64,64]

#指定ディレクトリの画像を取り込み
images = glob.glob('dataset/train2017/*')
images = random.sample(images,3000)
images_test = random.sample(images,500)
images_train = images

for im in images_test:
    images_train.remove(im)
    

#train画像取り込み
oriImg = cv2.imread(images_train[0])
oriImg = preprocess(oriImg)
oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
train_pcm_real,train_paf_real = get_heatmap_paf(oriImg)

pcm_real = cv2.resize(train_pcm_real, dsize=(picsize[0],picsize[1]))
paf_real = cv2.resize(train_paf_real, dsize=(picsize[0],picsize[1]))

pcm_real = pcm_real.reshape(1,picsize[0],picsize[1],19).astype("float32")
paf_real = paf_real.reshape(1,picsize[0],picsize[1],38).astype("float32")

pcm_real=pcm_real/Max
paf_real=paf_real/Max

np.save('dataset/pcm_train/{}_0.npy'.format(i),pcm_real)
np.save('dataset/paf_train/{}_0.npy'.format(i),paf_real)
#np.save('dataset/y/train_y_real_{}.npy'.format(i),y_real)
    
i += 1


for image in tqdm(images_train[1:]):
    
    oriImg = cv2.imread(image)
    oriImg = preprocess(oriImg)
    oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    
    pcm_real,paf_real = get_heatmap_paf(oriImg)
    
    pcm_real = cv2.resize(pcm_real, dsize=(picsize[0],picsize[1]))
    paf_real = cv2.resize(paf_real, dsize=(picsize[0],picsize[1]))

    pcm_real = pcm_real.reshape(1,picsize[0],picsize[1],19).astype("float32")
    paf_real = paf_real.reshape(1,picsize[0],picsize[1],38).astype("float32")
    
    pcm_real=pcm_real/Max
    paf_real=paf_real/Max
    
    np.save('dataset/pcm_train/{}_0.npy'.format(i),pcm_real)
    np.save('dataset/paf_train/{}_0.npy'.format(i),paf_real)
    #np.save('dataset/y/train_y_real_{}.npy'.format(i),y_real)
    
    i += 1

    #train_pcm_real = np.vstack((train_pcm_real,pcm_real))
    #train_paf_real = np.vstack((train_paf_real,paf_real))
    
    
#test画像取り込み
oriImg = cv2.imread(images_test[0])
oriImg = preprocess(oriImg)
oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
train_pcm_real,train_paf_real = get_heatmap_paf(oriImg)

pcm_real = cv2.resize(train_pcm_real, dsize=(picsize[0],picsize[1]))
paf_real = cv2.resize(train_paf_real, dsize=(picsize[0],picsize[1]))

pcm_real = pcm_real.reshape(1,picsize[0],picsize[1],19).astype("float32")
paf_real = paf_real.reshape(1,picsize[0],picsize[1],38).astype("float32")

pcm_real=pcm_real/Max
paf_real=paf_real/Max
    
np.save('dataset/pcm_test/{}_0.npy'.format(i),pcm_real)
np.save('dataset/paf_test/{}_0.npy'.format(i),paf_real)
#np.save('dataset/y/train_y_cmp_{}.npy'.format(i),y_cmp)

i += 1


for image in tqdm(images_test[1:]):
    
    oriImg = cv2.imread(image)
    oriImg = preprocess(oriImg)
    oriImg = cv2.resize(oriImg, dsize=(picsize_bf[0],picsize_bf[1]))
    
    pcm_real,paf_real = get_heatmap_paf(oriImg)

    pcm_real = cv2.resize(pcm_real, dsize=(picsize[0],picsize[1]))
    paf_real = cv2.resize(paf_real, dsize=(picsize[0],picsize[1]))
    
    pcm_real = pcm_real.reshape(1,picsize[0],picsize[1],19).astype("float32")
    paf_real = paf_real.reshape(1,picsize[0],picsize[1],38).astype("float32")
    
    pcm_real=pcm_real/Max
    paf_real=paf_real/Max
    
    np.save('dataset/pcm_test/{}_0.npy'.format(i),pcm_real)
    np.save('dataset/paf_test/{}_0.npy'.format(i),paf_real)
    
    i += 1
    
    
"""
for y in range(len(train_pcm_real)):
    
    train_y_real.append(0)

train_y_real = np.array(train_y_real)



np.save('train_pcm_cmp.npy',train_pcm_cmp)
np.save('train_paf_cmp.npy',train_paf_cmp)
np.save('train_y_cmp.npy',train_y_cmp)
#np.savetxt('train_pcm_real.csv',train_pcm_real,delimiter=',')
#np.savetxt('train_paf_rea.csv',train_paf_real,delimiter=',')
#np.savetxt('train_y_real.csv',train_y_real,delimiter=',')
"""