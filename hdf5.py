from __future__ import absolute_import
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import h5py

def load_data(hf5_data_path='data'):
    """
        read hdf5 files from specified path
        return: train_data, train_label, test_data, test_label
                shape: (num, height, width, channel)
    """
    train_path = os.path.join(hf5_data_path, 'train')
    test_path = os.path.join(hf5_data_path, 'test')
    if not os.path.exists(train_path):
        print(hf5_data_path, ' has no train directory')
        exit(1)
    if not os.path.exists(test_path):
        print(hf5_data_path, ' has no test directory')
        exit(1)

    train_data = []
    train_label = []
    train_dir = os.listdir(train_path)
    for file in train_dir:
        h5 = h5py.File(os.path.join(train_path, file), 'r')
        # print(h5['data'].name)
        # print(h5['data'].shape)
        # print(h5['data'].value)
        # val = h5['data'][:]
        train_data.append(h5['data'])
        train_label.append(h5['label'])

    test_data = []
    test_label = []
    test_dir = os.listdir(test_path)
    for file in test_dir:
        h5 = h5py.File(os.path.join(test_path, file), 'r')
        test_data.append(h5['data'])
        test_label.append(h5['label'])

    # print train_data.shape
    # print train_label.shape
    # print test_data.shape
    # print test_label.shape
    return train_data, train_label, test_data, test_label