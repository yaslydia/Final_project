import os

import keras

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score

import math
import random
from keras import optimizers
import numpy as np
import tensorflow as tf
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score

np.random.seed(0)

from keras.preprocessing import sequence
from keras import utils
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import GRU, Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, \
    GlobalAveragePooling2D
from keras.callbacks import History
from keras.models import Model, load_model

from collections import Counter

from sklearn.utils.class_weight import compute_class_weight

from myModel import build_model

import sys

sys.path.append("../..")
from loadData import *
from utils import *

batch_size = 200
n_ep = 2
fs = 200;
# half_size of the sliding window in samples
w_len = 8 * fs;
data_dim = w_len * 2
half_prec = 0.5
prec = 1
n_cl = 4

data_dir = '/home/cxyycl/scratch/Microsleep-code/data/files/'
f_set = '/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/file_sets_part2.mat'

create_tmp_dirs(
    ['/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/models/', '/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/'])

mat = spio.loadmat(f_set)

files_train = []
files_val = []
files_test = []

tmp = mat['files_train']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_train.extend(file)

tmp = mat['files_val']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_val.extend(file)
tmp = mat['files_test']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_test.extend(file)


def my_generator(data_train, targets_train, sample_list, shuffle=True):
    if shuffle:
        random.shuffle(sample_list)
    while True:
        for batch in batch_generator(sample_list, batch_size):
            batch_data1 = []
            batch_data2 = []
            batch_targets = []
            for sample in batch:
                [f, s, b, e, c] = sample
                sample_label = targets_train[f][c][s]
                sample_x1 = data_train[f][c][b:e + 1]
                sample_x2 = data_train[f][1][b:e + 1]
                sample_x = np.concatenate((sample_x1, sample_x2), axis=2)
                batch_data1.append(sample_x)
                batch_targets.append(sample_label)
            batch_data1 = np.stack(batch_data1, axis=0)
            batch_targets = np.array(batch_targets)
            batch_targets = to_categorical(batch_targets, n_cl)
            batch_data1 = (batch_data1) / 100
            batch_data1 = np.clip(batch_data1, -1, 1)
            yield [batch_data1], batch_targets


n_channels = 2

st0 = classes_global(data_dir, files_train)
cls = np.arange(n_cl)
cl_w = compute_class_weight(class_weight='balanced', classes=cls, y=st0)

(data_train, targets_train, N_samples) = load_data(data_dir, files_train, w_len)

N_batches = int(math.ceil((N_samples + 0.0) / batch_size))

(data_val, targets_val, N_samples_val) = load_data(data_dir, files_val, w_len)

(data_test, targets_test, N_samples_test) = load_data(data_dir,files_test, w_len)

# create indexes of samples
# each element is [file number in data_train, index in its targets, index of the beginning, index of the end of the window]
sample_list = []
for ch in range(2):
    for i in range(len(targets_train)):
        for j in range(len(targets_train[i][0])):
            mid = j * prec
            # we add the padding size
            mid += w_len
            wnd_begin = mid - w_len
            wnd_end = mid + w_len - 1
            sample_list.append([i, j, wnd_begin, wnd_end, ch])


sample_list_val = []

for i in range(len(targets_val)):
    sample_list_val.append([])
    for j in range(len(targets_val[i][0])):
        mid = j * prec
        # we add the padding size
        mid += w_len
        wnd_begin = mid - w_len
        wnd_end = mid + w_len - 1
        sample_list_val[i].append([i, j, wnd_begin, wnd_end, 0])
        
sample_list_test = []
for i in range(len(targets_test)):
	sample_list_test.append([])
	for j in range(len(targets_test[i][0])):
		mid = j*prec
		# we add the padding size
		mid += w_len
		wnd_begin = mid-w_len
		wnd_end = mid+w_len-1
		sample_list_test[i].append([i,j,wnd_begin, wnd_end, 0 ])
  
# print("Sample list test (first 5 samples):", sample_list_test[:5])

ordering = 'channels_last';
keras.backend.set_image_data_format(ordering)

[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)
Nadam = optimizers.Nadam()
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode=None)


###################################保存中间层特征和相应的标签######################################################

print("Start extracting features and labels...")

# 加载 cnn_eeg 模型
print("Loading Model......")
cnn_eeg = load_model('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/cnn_eeg_model2.h5')
 

generator_train = my_generator(data_train, targets_train, sample_list)
features = []  
labels = []  
for batch_index in range(N_batches):
    feature, label = next(generator_train)
    print("feature and lable length:",len(feature),len(label))
    cnn_eeg_train = cnn_eeg.predict(feature)
    features.append(cnn_eeg_train)
    labels.append(label)
features = np.vstack(features)  
labels = np.vstack(labels)  

print("Feature extraction finished.")
print("feature.shape:", features.shape)
print("label.shape:", labels.shape)

N_batches_val = int(math.ceil(N_samples_val / batch_size))
generator_val = my_generator(data_val, targets_val, sample_list_val[0], shuffle=False)
features_val = [] 
labels_val = []  
for batch_index in range(N_batches_val):
    feature_val, label_val = next(generator_val)
    cnn_eeg_val = cnn_eeg.predict(feature_val)
    features_val.append(cnn_eeg_val)
    labels_val.append(label_val)
features_val = np.vstack(features_val) 
labels_val = np.vstack(labels_val)  

print("feature_val.shape:", features_val.shape)
print("label_val.shape:", labels_val.shape)

N_batches_test = int(math.ceil(N_samples_test / batch_size))
generator_test = my_generator(data_test, targets_test, sample_list_test[0], shuffle=False)
features_test = []
labels_test = []
for batch_index in range(N_batches_test):
    feature_test, label_test = next(generator_test)
    cnn_eeg_test = cnn_eeg.predict(feature_test)  
    features_test.append(cnn_eeg_test)
    labels_test.append(label_test)
features_test = np.vstack(features_test)
labels_test = np.vstack(labels_test)

print("feature_test.shape:", features_test.shape)
print("label_test.shape:", labels_test.shape)

np.save('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/features_part2_0.npy', features)
np.save('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/labels_part2_0.npy', labels)
np.save('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/features_val_part2_0.npy', features_val)
np.save('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/labels_val_part2_0.npy', labels_val)
np.save('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/features_test_part2_0.npy', features_test)
np.save('/home/cxyycl/scratch/Microsleep-code/code/CNN_16s/set2/predictions/labels_test_part2_0.npy', labels_test)
##################################保存中间层特征和相应的标签######################################################