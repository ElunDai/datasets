#!/bin/python3

# -*- coding: utf-8 -*-
#==============================
#    Author: Elun Dai
#    Last modified: 2018-06-24 11:20
#    Filename: mnist.py
#    Description:
#    see:
#    https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
#=============================#
import numpy as np
import struct
import gzip
import array
import functools
import operator
import os
from ..utils import get_dataset
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

DIRECTORY = '/tmp/datasets/MNIST/'

def download_mnist(filename, directory=DIRECTORY):
    if os.path.exists(directory) is False:
        os.makedirs(directory)

    target = os.path.join(directory, filename)
    if os.path.exists(target) is False:
        dataset_url = 'http://yann.lecun.com/exdb/mnist/' + filename
        print('download:', dataset_url)
        urlretrieve(dataset_url, target)
        print('file have been saved to', target)

def get_mnist(directory=DIRECTORY):
    """Return train_images, train_labels, test_images, test_labels of MNIST dataset
    Parameters
    ----------
    directory : the directory contained MNIST dataset binary file

    Examples
    ----------
    X_train, y_train, X_test, y_test = get_mnist()
    """
#     for filename in ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte']:
#         file_path = os.path.join(directory, filename)
#         if os.path.exists(file_path): # binary file
#             fopen = open
#         else: # gzip file
#             filename = filename.replace('.', '-') + '.gz'
#             file_path = os.path.join(directory, filename)
#             fopen = gzip.open
#             if os.path.exists(file_path) is False:
#                 download_mnist(filename, directory)
        # read file
    fpaths = get_dataset(base_url='http://yann.lecun.com/exdb/mnist/',
                 filenames=['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'],
                 extract=False)

    mnist = list() 
    for fpath in fpaths:
        if os.path.exists(os.path.splitext(fpath)[0]): # binary file
            fopen = open
        else: # gzip file
            fopen = gzip.open
        with fopen(fpath, 'rb') as fd:
            mnist.append(parse_idx(fd))

    return mnist

def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https:tmpdocs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)
