# This code is for AIM2020 Raw2RGB, Full Resolution Raw image to RGB
# Input is (*,*,4) Raw image (PNG)
# Output is (*,*,3) RGB image (PNG)
# Author: Bingxin Hou
# 07/12/2020

import imageio
from skimage.color import lab2rgb

import numpy as np
import glob
import tensorflow.compat.v1 as tf

from keras.models import load_model
import keras.backend as K
import cv2
import os, sys
from PIL import Image

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

    test_data = np.float32(np.zeros((1, PATCH_WIDTH*10, PATCH_HEIGHT*15, 4)))

    for i in range(0, 1):
        I = np.asarray(imageio.imread(test_directory_phone))
        I = extract_bayer_channels(I)
        I=np.float32(cv2.resize(I, dsize=( PATCH_HEIGHT*15,PATCH_WIDTH*10,), interpolation=cv2.INTER_CUBIC))
        # I = I[ 0:PATCH_WIDTH*10+1,0:PATCH_HEIGHT*15+1]
        test_data[i, :] = I



    return test_data
def generate_patches(image_data,sub_patch_dim):

    patch_height = sub_patch_dim[0]
    patch_width = sub_patch_dim[1]
    x_spots =  list(range(0,image_data.shape[0] , patch_height))
    y_spots =  list(range(0, image_data.shape[1] , patch_width ))

    image_patches = []
    all_patches = []
    position =[]
    h=image_data.shape[0]
    w=image_data.shape[1]
    for x in x_spots:
        for y in y_spots:
            if ((x+patch_height)>image_data.shape[0]-1):
                x=image_data.shape[0]-1-patch_height
            if ((y+patch_width)>image_data.shape[1]-1):
                y=image_data.shape[1]-1-patch_width
            image_patches = image_data[x: x+patch_height,y: y+patch_width]
            all_patches.append(image_patches)
            position.append([x,y])
    all_patches = np.asarray(all_patches)
    position = np.array(position)
    return all_patches, position, h, w

def getData(input):


    y=load_test_data(input, 224, 224, 2)

    sub_patch_dim = [224, 224 ]

    y_patch, position1, h, w = generate_patches(np.squeeze(y), sub_patch_dim)
    y_patch1= np.float32(np.zeros((y_patch.shape[0], 448, 448, 4)))

    for i in range(y_patch.shape[0]):
        y_temp = np.float32(cv2.resize(y_patch[i], dsize=(448, 448), interpolation=cv2.INTER_CUBIC))
        y_patch1[i,:]=y_temp
    return y_patch1,  position1, h, w


input_path = os.path.join('AIM2020_ISP_fullres_test_raw/') # you need to set the input folder

side=448
mdl_path = 'C:/Users/houbi/Documents/research/Raw2RGB/FgSegNet_S/test_24X500_b5_ep15_Conv2_mse_LAB.h5' # you need to set the model path

for imageN in range(0,42):

    X_list = glob.glob(os.path.join(input_path, '*.png'))

    data, position, h, w = getData(X_list[imageN])

    position = np.asarray(position*2)


    lrelu = lambda xx: tf.keras.activations.relu(xx, alpha=0.1)

    model = load_model(mdl_path,compile=False,custom_objects={'<lambda>': lrelu})

    probs_patch = model.predict(data, batch_size=1, verbose=1)

    #### patches stitch to image ##############

    imagebank = np.zeros((h*2,w*2,3))
    ii=0
    patch = np.zeros(probs_patch.shape)
    patch = probs_patch
    patch[:,0:3,:,:]=probs_patch[:,3:6,:,:]
    patch[:,:,0:3,:]=probs_patch[:,:,3:6,:]
    patch[:,446:448,:,:]=probs_patch[:,442:444,:,:]
    patch[:,:,446:448,:]=probs_patch[:,:,442:444,:]


    for index in range(0, 150):
       # probs_patch1 =probs_patch[1:446,1:446,:], 1, 1, 1, 1, cv2.BORDER_CONSTANT)
       [startx,starty] = position[ii]
       imagebank[startx:startx+side,starty:starty+side] = patch[ii]
       ii=ii+1
    x = imagebank

    l = x[:, :, 0] * 50. + 50.
    a = (x[:, :, 1] * 128.)
    b = (x[:, :, 2] * 128.)
    #
    l = np.float32(np.expand_dims(l, axis=-1))
    a = np.float32(np.expand_dims(a, axis=-1))
    b = np.float32(np.expand_dims(b, axis=-1))

    x = lab2rgb(np.concatenate((l, a, b), axis=2))
    x = (x * 255.).astype('uint8')



    a = 'C:/Users/houbi/Documents/research/Raw2RGB/AIM2020_test/full_resolution/'+str(imageN)+'.png'  # you need to set the result image saving folder
    j = Image.fromarray(x)
    j.save(a, compress_level=0)

