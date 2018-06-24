#!/bin/python3

# -*- coding: utf-8 -*-
#==============================
#    Author: Elun Dai
#    Last modified: 2018-06-22 00:53
#    Filename: __init__.py
#    Description:
#    usage: from mnist import *
#           or
#           import mnist
#=============================#
# from .cifar import get_cifar100
from .cifar import *

__all__ = ['get_cifar100']

# train_images, train_labels, test_images, test_labels = mnist.get_mnist()
