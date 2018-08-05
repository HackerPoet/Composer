import numpy as np
import cv2
import os

def transpose_range(samples):
	merged_sample = np.zeros_like(samples[0])
	for sample in samples:
		merged_sample = np.maximum(merged_sample, sample)
	merged_sample = np.amax(merged_sample, axis=0)
	min_note = np.argmax(merged_sample)
	max_note = merged_sample.shape[0] - np.argmax(merged_sample[::-1])
	return min_note, max_note

def generate_add_centered_transpose(samples):
	num_notes = samples[0].shape[1]
	min_note, max_note = transpose_range(samples)
	s = num_notes/2 - (max_note + min_note)/2
	out_samples = samples
	out_lens = [len(samples), len(samples)]
	for i in xrange(len(samples)):
		out_sample = np.zeros_like(samples[i])
		out_sample[:,min_note+s:max_note+s] = samples[i][:,min_note:max_note]
		out_samples.append(out_sample)
	return out_samples, out_lens
	
def generate_all_transpose(samples, radius=6):
	num_notes = samples[0].shape[1]
	min_note, max_note = transpose_range(samples)
	min_shift = -min(radius, min_note)
	max_shift = min(radius, num_notes - max_note)
	out_samples = []
	out_lens = []
	for s in xrange(min_shift, max_shift):
		for i in xrange(len(samples)):
			out_sample = np.zeros_like(samples[i])
			out_sample[:,min_note+s:max_note+s] = samples[i][:,min_note:max_note]
			out_samples.append(out_sample)
		out_lens.append(len(samples))
	return out_samples, out_lens

def sample_to_pic(fname, sample, thresh=None):
	if thresh is not None:
		inverted = np.where(sample > thresh, 0, 1)
	else:
		inverted = 1.0 - sample
	cv2.imwrite(fname, inverted * 255)

def samples_to_pics(dir, samples, thresh=None):
	if not os.path.exists(dir): os.makedirs(dir)
	for i in xrange(samples.shape[0]):
		sample_to_pic(dir + '/s' + str(i) + '.png', samples[i], thresh)

def pad_songs(y, y_lens, max_len):
	y_shape = (y_lens.shape[0], max_len) + y.shape[1:]
	y_train = np.zeros(y_shape, dtype=np.float32)
	cur_ix = 0
	for i in xrange(y_lens.shape[0]):
		end_ix = cur_ix + y_lens[i]
		for j in xrange(max_len):
			k = j % (end_ix - cur_ix)
			y_train[i,j] = y[cur_ix + k]
		cur_ix = end_ix
	assert(end_ix == y.shape[0])
	return y_train

def sample_to_pattern(sample, ix, size):
	num_pats = 0
	pat_types = {}
	pat_list = []
	num_samples = len(sample) if type(sample) is list else sample.shape[0]
	for i in xrange(size):
		j = (ix + i) % num_samples
		measure = sample[j].tobytes()
		if measure not in pat_types:
			pat_types[measure] = num_pats
			num_pats += 1
		pat_list.append(pat_types[measure])
	return str(pat_list), pat_types

def embed_samples(samples):
	note_dict = {}
	n, m, p = samples.shape
	samples.flags.writeable = False
	e_samples = np.empty(samples.shape[:2], dtype=np.int32)
	for i in xrange(n):
		for j in xrange(m):
			note = samples[i,j].data
			if note not in note_dict:
				note_dict[note] = len(note_dict)
			e_samples[i,j] = note_dict[note]
	samples.flags.writeable = True
	lookup = np.empty((len(note_dict), p), dtype=np.float32)
	for k in note_dict:
		lookup[note_dict[k]] = k
	return e_samples, note_dict, lookup

def e_to_samples(e_samples, lookup):
	samples = np.empty(e_samples.shape + lookup.shape[-1:], dtype=np.float32)
	n, m = e_samples.shape
	for i in xrange(n):
		for j in xrange(m):
			samples[i,j] = lookup[e_samples[i,j]]
	return samples
	