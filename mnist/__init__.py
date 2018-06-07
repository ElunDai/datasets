#!/bin/python3

# -*- coding: utf-8 -*-
#==============================
#    Author: Elun Dai
#    Last modified: 2018-06-07 11:32
#    Filename: __init__.py
#    Description:
#    usage: from mnist import *
#           or
#           import mnist
#=============================#
from .mnist import get_mnist

__all__ = ['get_mnist']

# train_images, train_labels, test_images, test_labels = mnist.get_mnist()
