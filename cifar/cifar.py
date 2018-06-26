#!/bin/python3

# -*- coding: utf-8 -*-
#==============================
#    Author: Elun Dai
#    Last modified: 2018-06-25 01:06
#    Filename: cifar.py
#    Description:
#    
#=============================#
import os
import numpy as np
import platform
import pickle
from ..utils import downloader

DIRECTORY = '/tmp/datasets/cifar'

# fine label to coarse label
CIFAR100_LABEL_DECT = dict([(4, 0), (30, 0), (55, 0), (72, 0), (95, 0),
 (1, 1), (32, 1), (67, 1), (73, 1), (91, 1),
 (54, 2), (62, 2), (70, 2), (82, 2), (92, 2),
 (9, 3), (10, 3), (16, 3), (28, 3), (61, 3),
 (0, 4), (51, 4), (53, 4), (57, 4), (83, 4),
 (22, 5), (39, 5), (40, 5), (86, 5), (87, 5),
 (5, 6), (20, 6), (25, 6), (84, 6), (94, 6),
 (6, 7), (7, 7), (14, 7), (18, 7), (24, 7),
 (3, 8), (42, 8), (43, 8), (88, 8), (97, 8),
 (12, 9), (17, 9), (37, 9), (68, 9), (76, 9),
 (23, 10), (33, 10), (49, 10), (60, 10), (71, 10),
 (15, 11), (19, 11), (21, 11), (31, 11), (38, 11),
 (34, 12), (63, 12), (64, 12), (66, 12), (75, 12),
 (26, 13), (45, 13), (77, 13), (79, 13), (99, 13),
 (2, 14), (11, 14), (35, 14), (46, 14), (98, 14),
 (27, 15), (29, 15), (44, 15), (78, 15), (93, 15),
 (36, 16), (50, 16), (65, 16), (74, 16), (80, 16),
 (47, 17), (52, 17), (56, 17), (59, 17), (96, 17),
 (8, 18), (13, 18), (48, 18), (58, 18), (90, 18),
 (41, 19), (69, 19), (81, 19), (85, 19), (89, 19)])

def load_pickle(filename):
    """parse_pickle(filename):
    return a dict.

    arguements
    ----------
    filename: the pickle file path.
    """
    version = platform.python_version_tuple()
    with open(filename, 'rb') as f:
        if version[0] == '2':
            return  pickle.load(f)
        elif version[0] == '3':
            return  pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))


def get_cifar100(directory=DIRECTORY, channel="rgb"):
    """get_cifar100(directory=DIRECTORY):
    Get the cifar100 dataset.

    arguements
    ----------
    directory : the directory contained CIFAR100 dataset python pickle.
    channel: 'rgb' by default, and you can set to 'bgr'

    return values
    ---------
    meta: include fine_label_names and coarse_label_names.
    X_train: 10000 train datas with shape 32 X 32 X 3.
    y_train_fl: 10000 train fine labels.
    X_train: 10000 train datas with shape 32 X 32 X 3.
    y_train_fl: 10000 train fine labels.

    examples
    ----------
    meta, X_train, y_train,  X_test, y_test = get_cifar100()
    """
    url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    downloader.get_dataset(urls=url, directory=directory, extract=True)
    fdir = os.path.join(directory, 'cifar-100-python/')
    meta = load_pickle(os.path.join(fdir, 'meta'))

    train = load_pickle(os.path.join(fdir, 'train'))
    test = load_pickle(os.path.join(fdir, 'test'))
    X_train = train.get('data').reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    if channel is not 'rgb':
        X_train = X_train[:, :, :, ::-1] # RGB to BGR
    y_train_fl = train.get('fine_labels')
#     y_train_cl = train.get('coarse_labels')

    test = load_pickle(os.path.join(fdir, 'test'))
    X_test = test.get('data').reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    if channel is not 'rgb':
        X_test = X_test[:, :, :, ::-1] # RGB to BGR
    y_test_fl = test.get('fine_labels')
#     y_test_cl = test.get('coarse_labels')

    return (meta, X_train, y_train_fl, X_test, y_test_fl)

def get_label(fine_label_idx, meta=None):
    """get_label(fine_label_idx, meta=None)
    if meta isn't None, it return (coarse label names, fine label names).
    otherwise it only return the index of couarse label,
    """
    coarse_label_idx = CIFAR100_LABEL_DECT.get(fine_label_idx)
    if meta is not None:
        return (meta.get('coarse_label_names')[coarse_label_idx],
                meta.get('fine_label_names')[fine_label_idx])
    else:
        return coarse_label_idx
