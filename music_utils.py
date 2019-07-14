#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utils to edit music.
"""

import numpy as np


def find_sample_range(samples):
    """
    Find sample range.
    :param samples:
    :return:
    """

    # merge all samples
    merged_sample = np.zeros_like(samples[0])
    for sample in samples:
        merged_sample = np.maximum(merged_sample, sample)

    # get all pitches being played
    merged_sample = np.amax(merged_sample, axis=0)

    # get min and max note
    min_note = np.argmax(merged_sample)
    max_note = merged_sample.shape[0] - np.argmax(merged_sample[::-1])
    return min_note, max_note


def generate_centered_transpose(samples):
    """
    Center samples towards the middle of the pitch range.
    :param samples:
    :return:
    """
    num_notes = samples[0].shape[1]
    min_note, max_note = find_sample_range(samples)

    # find deviation from pitch center
    center_deviation = num_notes / 2 - (max_note + min_note) / 2
    out_samples = samples
    out_lengths = [len(samples), len(samples)]

    # center every sample by moving it by center_deviation
    for i in range(len(samples)):
        out_sample = np.zeros_like(samples[i])
        out_sample[:, min_note + int(center_deviation):max_note + int(center_deviation)] = samples[i][:, min_note:max_note]
        out_samples.append(out_sample)
    return out_samples, out_lengths
