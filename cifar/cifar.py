#!/bin/python3

# -*- coding: utf-8 -*-
#==============================
#    Author: Elun Dai
#    Last modified: 2018-06-24 09:44
#    Filename: cifar.py
#    Description:
#    
#=============================#
import os
import numpy as np
from ..utils import downloader

DIRECTORY='/tmp/datasets/cifar100'


def get_cifar100(directory=DIRECTORY): 
    url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    downloader.get_dataset(urls=url, directory=directory, extract=True)
