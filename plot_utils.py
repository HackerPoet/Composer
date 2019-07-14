#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot utility functions.
"""

import os
import cv2
import numpy as np


def plot_sample(file_name, sample, threshold=None):
    if threshold is not None:
        inverted = np.where(sample > threshold, 0, 1)
    else:
        inverted = 1.0 - sample
    cv2.imwrite(file_name, inverted * 255)


def plot_samples(folder, samples, threshold=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(samples.shape[0]):
        plot_sample(folder + '/s' + str(i) + '.png', samples[i], threshold)
