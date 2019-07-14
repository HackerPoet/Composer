#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load songs from midi, preprocess and save them in numpy format.
"""

import midi_utils
import os
import music_utils
import numpy as np
import argparse


def preprocess_songs(data_folders):
    """
    Load and preprocess the songs from the data folders and turn them into a dataset of samples/pitches and lengths of the tones.
    :param data_folders:
    :return:
    """

    all_samples = []
    all_lengths = []

    # keep some statistics
    succeeded = 0
    failed = 0
    ignored = 0

    # load songs
    print("Loading songs...")
    # walk folders and look for midi files
    for folder in data_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)
                if not (path.endswith('.mid') or path.endswith('.midi')):
                    continue

                # turn midi into samples
                try:
                    samples = midi_utils.midi_to_samples(path)
                except Exception as e:
                    print("ERROR ", path)
                    print(e)
                    failed += 1
                    continue

                # if the midi does not produce the minimal number of sample/measures, we skip it
                if len(samples) < 16:
                    print('WARN', path, 'Sample too short, unused')
                    ignored += 1
                    continue

                # transpose samples (center them in full range to get more training samples for the same tones)
                samples, lengths = music_utils.generate_centered_transpose(samples)
                all_samples += samples
                all_lengths += lengths
                print('SUCCESS', path, len(samples), 'samples')
                succeeded += 1

    assert (sum(all_lengths) == len(all_samples))  # assert equal number of samples and lengths

    # save all to disk
    print("Saving " + str(len(all_samples)) + " samples...")
    all_samples = np.array(all_samples, dtype=np.uint8)  # reduce size when saving
    all_lengths = np.array(all_lengths, dtype=np.uint32)
    np.save('data/interim/samples.npy', all_samples)
    np.save('data/interim/lengths.npy', all_lengths)
    print('Done: ', succeeded, 'succeded,', ignored, 'ignored,', failed, 'failed of', succeeded + ignored + failed, 'in total')


if __name__ == "__main__":
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Load songs, preprocess them and put them into a dataset.')
    parser.add_argument('--data_folder', default=["data/raw"], type=str, help='The path to the midi data', action='append')

    args = parser.parse_args()
    preprocess_songs(args.data_folder)
