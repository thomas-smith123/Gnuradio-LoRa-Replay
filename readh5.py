import h5py
import numpy as np
import os

file_path = 'dataset_training_no_aug.h5'
f = h5py.File(file_path, 'r')
label_t = f['label'][:]-1
data_t = f['data'][:]

# a, b =  data_t.shape
# data_t = data_t[1, :b//2]+1j*data_t[1, b//2:]

np.save('dataset/label.npy', label_t)
np.save('dataset/data.npy', data_t)
pp = np.fromfile('save_test', dtype=np.complex64)
pass