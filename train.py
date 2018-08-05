import sys, random, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2
import util
import midi

NUM_EPOCHS = 2000
LR = 0.001
CONTINUE_TRAIN = False
PLAY_ONLY = False
USE_EMBEDDING = False
USE_VAE = False
WRITE_HISTORY = True
NUM_RAND_SONGS = 10
DO_RATE = 0.1
BN_M = 0.9
VAE_B1 = 0.02
VAE_B2 = 0.1

BATCH_SIZE = 350
MAX_LENGTH = 16
PARAM_SIZE = 120
NUM_OFFSETS = 16 if USE_EMBEDDING else 1

def plotScores(scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	plt.plot(scores)
	plt.ylim([0.0, 0.009])
	plt.xlabel('Epoch')
	loc = ('upper right' if on_top else 'lower right')
	plt.draw()
	plt.savefig(fname)

def save_config():
	with open('config.txt', 'w') as fout:
		fout.write('LR:          ' + str(LR) + '\n')
		fout.write('BN_M:        ' + str(BN_M) + '\n')
		fout.write('BATCH_SIZE:  ' + str(BATCH_SIZE) + '\n')
		fout.write('NUM_OFFSETS: ' + str(NUM_OFFSETS) + '\n')
		fout.write('DO_RATE:     ' + str(DO_RATE) + '\n')
		fout.write('num_songs:   ' + str(num_songs) + '\n')
		fout.write('optimizer:   ' + type(model.optimizer).__name__ + '\n')

###################################
#  Load Keras
###################################
print "Loading Keras..."
import os, math
os.environ['THEANORC'] = "./gpu.theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__

import keras
print "Keras Version: " + keras.__version__
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Permute, RepeatVector, ActivityRegularization, TimeDistributed, Lambda, SpatialDropout1D
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.initializers import RandomNormal
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
K.set_image_data_format('channels_first')

#Fix the random seed so that training comparisons are easier to make
np.random.seed(0)
random.seed(0)

if WRITE_HISTORY:
	#Create folder to save models into
	if not os.path.exists('History'):
		os.makedirs('History')

###################################
#  Load Dataset
###################################
print "Loading Data..."
y_samples = np.load('samples.npy')
y_lengths = np.load('lengths.npy')
num_samples = y_samples.shape[0]
num_songs = y_lengths.shape[0]
print "Loaded " + str(num_samples) + " samples from " + str(num_songs) + " songs."
print np.sum(y_lengths)
assert(np.sum(y_lengths) == num_samples)

print "Padding Songs..."
x_shape = (num_songs * NUM_OFFSETS, 1)
y_shape = (num_songs * NUM_OFFSETS, MAX_LENGTH) + y_samples.shape[1:]
x_orig = np.expand_dims(np.arange(x_shape[0]), axis=-1)
y_orig = np.zeros(y_shape, dtype=y_samples.dtype)
cur_ix = 0
for i in xrange(num_songs):
	for ofs in xrange(NUM_OFFSETS):
		ix = i*NUM_OFFSETS + ofs
		end_ix = cur_ix + y_lengths[i]
		for j in xrange(MAX_LENGTH):
			k = (j + ofs) % (end_ix - cur_ix)
			y_orig[ix,j] = y_samples[cur_ix + k]
	cur_ix = end_ix
assert(end_ix == num_samples)
x_train = np.copy(x_orig)
y_train = np.copy(y_orig)

def to_song(encoded_output):
	return np.squeeze(decoder([np.round(encoded_output), 0])[0])

def reg_mean_std(x):
	s = K.log(K.sum(x * x))
	return s*s

def vae_sampling(args):
	z_mean, z_log_sigma_sq = args
	epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=VAE_B1)
	return z_mean + K.exp(z_log_sigma_sq * 0.5) * epsilon

def vae_loss(x, x_decoded_mean):
	xent_loss = binary_crossentropy(x, x_decoded_mean)
	kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
	return xent_loss - kl_loss
	
test_ix = 0
y_test_song = np.copy(y_train[test_ix:test_ix+1])
x_test_song = np.copy(x_train[test_ix:test_ix+1])
midi.samples_to_midi(y_test_song[0], 'gt.mid', 16)

###################################
#  Create Model
###################################
if CONTINUE_TRAIN or PLAY_ONLY:
	print "Loading Model..."
	model = load_model('model.h5', custom_objects=custom_objects)
else:
	print "Building Model..."

	if USE_EMBEDDING:
		x_in = Input(shape=x_shape[1:])
		print (None,) + x_shape[1:]
		x = Embedding(x_train.shape[0], PARAM_SIZE, input_length=1)(x_in)
		x = Flatten(name='pre_encoder')(x)
	else:
		x_in = Input(shape=y_shape[1:])
		print (None,) + y_shape[1:]
		x = Reshape((y_shape[1], -1))(x_in)
		print K.int_shape(x)
		
		x = TimeDistributed(Dense(2000, activation='relu'))(x)
		print K.int_shape(x)
		
		x = TimeDistributed(Dense(200, activation='relu'))(x)
		print K.int_shape(x)

		x = Flatten()(x)
		print K.int_shape(x)

		x = Dense(1600, activation='relu')(x)
		print K.int_shape(x)
		
		if USE_VAE:
			z_mean = Dense(PARAM_SIZE)(x)
			z_log_sigma_sq = Dense(PARAM_SIZE)(x)
			x = Lambda(vae_sampling, output_shape=(PARAM_SIZE,), name='pre_encoder')([z_mean, z_log_sigma_sq])
		else:
			x = Dense(PARAM_SIZE)(x)
			x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
	print K.int_shape(x)
	
	x = Dense(1600, name='encoder')(x)
	x = BatchNormalization(momentum=BN_M)(x)
	x = Activation('relu')(x)
	if DO_RATE > 0:
		x = Dropout(DO_RATE)(x)
	print K.int_shape(x)

	x = Dense(MAX_LENGTH * 200)(x)
	print K.int_shape(x)
	x = Reshape((MAX_LENGTH, 200))(x)
	x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
	x = Activation('relu')(x)
	if DO_RATE > 0:
		x = Dropout(DO_RATE)(x)
	print K.int_shape(x)

	x = TimeDistributed(Dense(2000))(x)
	x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
	x = Activation('relu')(x)
	if DO_RATE > 0:
		x = Dropout(DO_RATE)(x)
	print K.int_shape(x)

	x = TimeDistributed(Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
	print K.int_shape(x)
	x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x)
	print K.int_shape(x)
	
	if USE_VAE:
		model = Model(x_in, x)
		model.compile(optimizer=Adam(lr=LR), loss=vae_loss)
	else:
		model = Model(x_in, x)
		model.compile(optimizer=RMSprop(lr=LR), loss='binary_crossentropy')

	plot_model(model, to_file='model.png', show_shapes=True)

###################################
#  Train
###################################
print "Compiling SubModels..."
func = K.function([model.get_layer('encoder').input, K.learning_phase()],
				  [model.layers[-1].output])
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, PARAM_SIZE))
np.save('rand.npy', rand_vecs)

def make_rand_songs(write_dir, rand_vecs):
	for i in xrange(rand_vecs.shape[0]):
		x_rand = rand_vecs[i:i+1]
		y_song = func([x_rand, 0])[0]
		midi.samples_to_midi(y_song[0], write_dir + 'rand' + str(i) + '.mid', 16, 0.25)

def make_rand_songs_normalized(write_dir, rand_vecs):
	if USE_EMBEDDING:
		x_enc = np.squeeze(enc.predict(x_orig))
	else:
		x_enc = np.squeeze(enc.predict(y_orig))
	
	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	u, s, v = np.linalg.svd(x_cov)
	e = np.sqrt(s)

	print "Means: ", x_mean[:6]
	print "Evals: ", e[:6]
	
	np.save(write_dir + 'means.npy', x_mean)
	np.save(write_dir + 'stds.npy', x_stds)
	np.save(write_dir + 'evals.npy', e)
	np.save(write_dir + 'evecs.npy', v)

	x_vecs = x_mean + np.dot(rand_vecs * e, v)
	make_rand_songs(write_dir, x_vecs)
	
	title = ''
	if '/' in write_dir:
		title = 'Epoch: ' + write_dir.split('/')[-2][1:]
	
	plt.clf()
	e[::-1].sort()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), e, align='center')
	plt.draw()
	plt.savefig(write_dir + 'evals.png')

	plt.clf()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), x_mean, align='center')
	plt.draw()
	plt.savefig(write_dir + 'means.png')
	
	plt.clf()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), x_stds, align='center')
	plt.draw()
	plt.savefig(write_dir + 'stds.png')

if PLAY_ONLY:
	print "Generating Songs..."
	make_rand_songs_normalized('', rand_vecs)
	for i in xrange(20):
		x_test_song = x_train[i:i+1]
		y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
		midi.samples_to_midi(y_song, 'gt' + str(i) + '.mid', 16)
	exit(0)
		  
print "Training..."
save_config()
train_loss = []
ofs = 0

for iter in xrange(NUM_EPOCHS):
	if USE_EMBEDDING:
		history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
	else:
		cur_ix = 0
		for i in xrange(num_songs):
			end_ix = cur_ix + y_lengths[i]
			for j in xrange(MAX_LENGTH):
				k = (j + ofs) % (end_ix - cur_ix)
				y_train[i,j] = y_samples[cur_ix + k]
			cur_ix = end_ix
		assert(end_ix == num_samples)
		ofs += 1

		history = model.fit(y_train, y_train, batch_size=BATCH_SIZE, epochs=1)

	loss = history.history["loss"][-1]
	train_loss.append(loss)
	print "Train Loss: " + str(train_loss[-1])
	
	if WRITE_HISTORY:
		plotScores(train_loss, 'History/Scores.png', True)
	else:
		plotScores(train_loss, 'Scores.png', True)
	
	i = iter + 1
	if i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450] or (i % 100 == 0):
		write_dir = ''
		if WRITE_HISTORY:
			#Create folder to save models into
			write_dir = 'History/e' + str(i)
			if not os.path.exists(write_dir):
				os.makedirs(write_dir)
			write_dir += '/'
			model.save('History/model.h5')
		else:
			model.save('model.h5')
		print "Saved"

		if USE_EMBEDDING:
			y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
		else:
			y_song = model.predict(y_test_song, batch_size=BATCH_SIZE)[0]
		util.samples_to_pics(write_dir + 'test', y_song)
		midi.samples_to_midi(y_song, write_dir + 'test.mid', 16)

		make_rand_songs_normalized(write_dir, rand_vecs)

print "Done"
