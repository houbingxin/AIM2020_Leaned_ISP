#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code is for AIM2020 Raw2RGB, 224*224 Raw image to RGB
# Input is 224*224*4 Raw image (PNG)
# Output is 224*224*3 RGB image (PNG)
# Author: Bingxin Hou
# 07/12/2020
from PIL import Image
import keras
import scipy
from PIL import Image
import imageio
from skimage.color import rgb2lab, lab2rgb

import numpy as np
import os, glob
import PIL
import tensorflow.compat.v1 as tf

from keras.models import load_model
import keras.backend as K
import cv2
import os, sys
from PIL import Image

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print('Elapsed time is'+ str(time.time() - startTime_for_tictoc) + 'seconds')
    else:
        print('Toc: start time not set')

cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

def extract_bayer_channels(raw):
    # Reshape the input bayer image

    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


def load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):
    # test_directory_dslr = dataset_dir + 'test/canon/'
    test_directory_phone = dataset_dir # + 'test/huawei_raw/'

    # NUM_TEST_IMAGES = 1204
    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.float32(np.zeros((NUM_TEST_IMAGES, PATCH_WIDTH*2, PATCH_HEIGHT*2, 4)))
    test_answ = np.float32(np.zeros((NUM_TEST_IMAGES, int(PATCH_WIDTH * DSLR_SCALE), int(PATCH_HEIGHT * DSLR_SCALE), 3)))

    for i in range(0, NUM_TEST_IMAGES):
        I = np.asarray(imageio.imread((test_directory_phone + str(i) + '.png')))
        I = extract_bayer_channels(I)
        I=np.float32(cv2.resize(I, dsize=(448, 448), interpolation=cv2.INTER_CUBIC))

        test_data[i, :] = I



    return test_data

#
def getData(input_path):


    X = []
    X_name = []

    X_list= glob.glob(os.path.join(input_path, '*.png'))

    for i in range(0,len(X_list)):

        x_name = X_list[i].split('.')[0]
        x_name = x_name.split('\\')[1]

        x = cv2.imread(X_list[i], cv2.IMREAD_UNCHANGED)

        X.append(x)
        X_name.append(x_name)

    XX = np.asarray(X)
    XX_name = np.array(X_name)

    return XX, XX_name

input_path = os.path.join('AIM2020_ISP_test_raw/') # you need to set the input folder


data_no,name = getData(input_path)

data =  load_test_data(input_path,  224, 224, 2)
lrelu= lambda xx: tf.keras.activations.relu(xx, alpha=0.1)

num_in_row = 1
num_in_col = 2
mdl_path = 'C:/Users/houbi/Documents/research/Raw2RGB/FgSegNet_S/test_24X500_b5_ep15_Conv2_mse_LAB.h5' # you need to set the model path

model = load_model(mdl_path,compile=False,custom_objects={'<lambda>': lrelu})
#print(get_flops(model))
print(model.summary())
tic()
probs = model.predict(data, batch_size=1, verbose=1)
toc()

for frame_idx in range(0,len(probs)): # display frame index

    x = probs[frame_idx]

    #############LAB version###################33

    l = x[:, :, 0] * 50.+50.
    a = (x[:, :, 1] * 128.)
    b = (x[:, :, 2] * 128.)
    #
    l = np.float32(np.expand_dims(l, axis=-1))
    a = np.float32(np.expand_dims(a, axis=-1))
    b = np.float32(np.expand_dims(b, axis=-1))

    x = lab2rgb(np.concatenate((l,a,b),axis=2))
    x = (x*255.).astype('uint8')


    output_path = 'C:/Users/houbi/Documents/research/Raw2RGB/TestingPhone_RGB2/'+ str(frame_idx)+'.png'  # you need to set the result image saving folder

    j = Image.fromarray(x)
    j.save(output_path, compress_level=0)

