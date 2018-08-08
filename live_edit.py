import pygame
import random, math
import numpy as np
import cv2
import pyaudio
import midi
import wave

#User constants
device = "cpu"
dir_name = 'History/'
sub_dir_name = 'e2000/'
sample_rate = 48000
note_dt = 2000        #Num Samples
note_duration = 20000 #Num Samples
note_decay = 5.0 / sample_rate
num_params = 120
num_measures = 16
num_sigmas = 5.0
note_thresh = 32
use_pca = True
is_ae = True

background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)]

note_w = 96
note_h = 96
note_pad = 2

notes_rows = num_measures / 8
notes_cols = 8

slider_num = min(40, num_params)
slider_h = 200
slider_pad = 5
tick_pad = 4

control_w = 210
control_h = 30
control_pad = 5
control_num = 3
control_colors = [(255,0,0), (0,255,0), (0,0,255)]
control_inits = [0.75, 0.5, 0.5]

#Derived constants
notes_w = notes_cols * (note_w + note_pad*2)
notes_h = notes_rows * (note_h + note_pad*2)
sliders_w = notes_w
sliders_h = slider_h + slider_pad*2
controls_w = control_w * control_num
controls_h = control_h
window_w = notes_w
window_h = notes_h + sliders_h + controls_h
slider_w = (window_w - slider_pad*2) / slider_num
notes_x = 0
notes_y = sliders_h
sliders_x = slider_pad
sliders_y = slider_pad
controls_x = (window_w - controls_w) / 2
controls_y = notes_h + sliders_h

#Global variables
prev_mouse_pos = None
mouse_pressed = 0
cur_slider_ix = 0
cur_control_ix = 0
volume = 3000
instrument = 0
needs_update = True
cur_params = np.zeros((num_params,), dtype=np.float32)
cur_notes = np.zeros((num_measures, note_h, note_w), dtype=np.uint8)
cur_controls = np.array(control_inits, dtype=np.float32)

#Setup audio stream
audio = pyaudio.PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False
def audio_callback(in_data, frame_count, time_info, status):
	global audio_time
	global audio_notes
	global audio_reset
	global note_time
	global note_time_dt

	#Check if needs restart
	if audio_reset:
		audio_notes = []
		audio_time = 0
		note_time = 0
		note_time_dt = 0
		audio_reset = False
	
	#Check if paused
	if audio_pause and status is not None:
		data = np.zeros((frame_count,), dtype=np.float32)
		return (data.tobytes(), pyaudio.paContinue)
	
	#Find and add any notes in this time window
	cur_dt = note_dt
	while note_time_dt < audio_time + frame_count:
		measure_ix = note_time / note_h
		if measure_ix >= num_measures:
			break
		note_ix = note_time % note_h
		notes = np.where(cur_notes[measure_ix, note_ix] >= note_thresh)[0]
		for note in notes:
			freq = 2 * 38.89 * pow(2.0, note / 12.0) / sample_rate
			audio_notes.append((note_time_dt, freq))
		note_time += 1
		note_time_dt += cur_dt
			
	#Generate the tones
	data = np.zeros((frame_count,), dtype=np.float32)
	for t,f in audio_notes:
		x = np.arange(audio_time - t, audio_time + frame_count - t)
		x = np.maximum(x, 0)

		if instrument == 0:
			w = np.sign(1 - np.mod(x * f, 2))            #Square
		elif instrument == 1:
			w = np.mod(x * f - 1, 2) - 1                 #Sawtooth
		elif instrument == 2:
			w = 2*np.abs(np.mod(x * f - 0.5, 2) - 1) - 1 #Triangle
		elif instrument == 3:
			w = np.sin(x * f * math.pi)                  #Sine
		
		#w = np.floor(w*8)/8
		w[x == 0] = 0
		w *= volume * np.exp(-x*note_decay)
		data += w
	data = np.clip(data, -32000, 32000).astype(np.int16)

	#Remove notes that are too old
	audio_time += frame_count
	audio_notes = [(t,f) for t,f in audio_notes if audio_time < t + note_duration]
	
	#Reset if loop occurs
	if note_time / note_h >= num_measures:
		audio_time = 0
		note_time = 0
		note_time_dt = 0
		audio_notes = []
	
	#Return the sound clip
	return (data.tobytes(), pyaudio.paContinue)
	
#Keras
print "Loading Keras..."
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
import keras
print "Keras Version: " + keras.__version__
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.local import LocallyConnected2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

print "Loading Encoder..."
model = load_model(dir_name + 'model.h5')
enc = K.function([model.get_layer('encoder').input, K.learning_phase()],
				 [model.layers[-1].output])
enc_model = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

print "Loading Statistics..."
means = np.load(dir_name + sub_dir_name + 'means.npy')
evals = np.load(dir_name + sub_dir_name + 'evals.npy')
evecs = np.load(dir_name + sub_dir_name + 'evecs.npy')
stds = np.load(dir_name + sub_dir_name + 'stds.npy')

print "Loading Songs..."
y_samples = np.load('samples.npy')
y_lengths = np.load('lengths.npy')

#Open a window
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((window_w, window_h))
notes_surface = screen.subsurface((notes_x, notes_y, notes_w, notes_h))
pygame.display.set_caption('MusicEdit')
font = pygame.font.SysFont("monospace", 15)

#Start the audio stream
audio_stream = audio.open(
	format=audio.get_format_from_width(2),
	channels=1,
	rate=sample_rate,
	output=True,
	stream_callback=audio_callback)
audio_stream.start_stream()

def update_mouse_click(mouse_pos):
	global cur_slider_ix
	global cur_control_ix
	global mouse_pressed
	x = (mouse_pos[0] - sliders_x)
	y = (mouse_pos[1] - sliders_y)

	if x >= 0 and y >= 0 and x < sliders_w and y < sliders_h:
		cur_slider_ix = x / slider_w
		mouse_pressed = 1
		
	x = (mouse_pos[0] - controls_x)
	y = (mouse_pos[1] - controls_y)
	if x >= 0 and y >= 0 and x < controls_w and y < controls_h:
		cur_control_ix = x / control_w
		mouse_pressed = 2

def apply_controls():
	global note_thresh
	global note_dt
	global volume

	note_thresh = (1.0 - cur_controls[0]) * 200 + 10
	note_dt = (1.0 - cur_controls[1]) * 1800 + 200
	volume = cur_controls[2] * 6000
		
def update_mouse_move(mouse_pos):
	global needs_update

	if mouse_pressed == 1:
		y = (mouse_pos[1] - sliders_y)
		if y >= 0 and y <= slider_h:
			val = (float(y) / slider_h - 0.5) * (num_sigmas * 2)
			cur_params[cur_slider_ix] = val
			needs_update = True
	elif mouse_pressed == 2:
		x = (mouse_pos[0] - (controls_x + cur_control_ix*control_w))
		if x >= control_pad and x <= control_w - control_pad:
			val = float(x - control_pad) / (control_w - control_pad*2)
			cur_controls[cur_control_ix] = val
			apply_controls()

def draw_controls():
	for i in xrange(control_num):
		x = controls_x + i * control_w + control_pad
		y = controls_y + control_pad
		w = control_w - control_pad*2
		h = control_h - control_pad*2
		col = control_colors[i]

		pygame.draw.rect(screen, col, (x, y, int(w*cur_controls[i]), h))
		pygame.draw.rect(screen, (0,0,0), (x, y, w, h), 1)
		
def draw_sliders():
	for i in xrange(slider_num):
		slider_color = slider_colors[i % len(slider_colors)]
		x = sliders_x + i * slider_w
		y = sliders_y

		cx = x + slider_w / 2
		cy_1 = y
		cy_2 = y + slider_h
		pygame.draw.line(screen, slider_color, (cx, cy_1), (cx, cy_2))
		
		cx_1 = x + tick_pad
		cx_2 = x + slider_w - tick_pad
		for j in xrange(int(num_sigmas * 2 + 1)):
			ly = y + slider_h/2.0 + (j-num_sigmas)*slider_h/(num_sigmas*2.0)
			ly = int(ly)
			col = (0,0,0) if j - num_sigmas == 0 else slider_color
			pygame.draw.line(screen, col, (cx_1, ly), (cx_2, ly))
			
		py = y + int((cur_params[i] / (num_sigmas * 2) + 0.5) * slider_h)
		pygame.draw.circle(screen, slider_color, (cx, py), (slider_w - tick_pad)/2)

def notes_to_img(notes):
	output = np.full((3, notes_h, notes_w), 64, dtype=np.uint8)

	for i in xrange(notes_rows):
		for j in xrange(notes_cols):
			x = note_pad + j*(note_w + note_pad*2)
			y = note_pad + i*(note_h + note_pad*2)
			ix = i*notes_cols + j

			measure = np.rot90(notes[ix])
			played_only = np.where(measure >= note_thresh, 255, 0)
			output[0,y:y+note_h,x:x+note_w] = np.minimum(measure * (255.0 / note_thresh), 255.0)
			output[1,y:y+note_h,x:x+note_w] = played_only
			output[2,y:y+note_h,x:x+note_w] = played_only

	return np.transpose(output, (2, 1, 0))
			
def draw_notes():
	pygame.surfarray.blit_array(notes_surface, notes_to_img(cur_notes))

	measure_ix = note_time / note_h
	note_ix = note_time % note_h
	x = notes_x + note_pad + (measure_ix % notes_cols) * (note_w + note_pad*2) + note_ix
	y = notes_y +note_pad + (measure_ix / notes_cols) * (note_h + note_pad*2)
	pygame.draw.rect(screen, (255,255,0), (x, y, 4, note_h), 0)
	
#Main loop
running = True
rand_ix = 0
cur_len = 0
apply_controls()
while running:
	#Process events
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
			break
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				prev_mouse_pos = pygame.mouse.get_pos()
				update_mouse_click(prev_mouse_pos)
				update_mouse_move(prev_mouse_pos)
			elif pygame.mouse.get_pressed()[2]:
				cur_params = np.zeros((num_params,), dtype=np.float32)
				needs_update = True
		elif event.type == pygame.MOUSEBUTTONUP:
			mouse_pressed = 0
			prev_mouse_pos = None
		elif event.type == pygame.MOUSEMOTION and mouse_pressed > 0:
			update_mouse_move(pygame.mouse.get_pos())
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_r:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -num_sigmas, num_sigmas)
				needs_update = True
				audio_reset = True
			if event.key == pygame.K_e:
				cur_params = np.clip(np.random.normal(0.0, 2.0, (num_params,)), -num_sigmas, num_sigmas)
				needs_update = True
				audio_reset = True
			if event.key == pygame.K_o:
				print "RandIx: " + str(rand_ix)
				if is_ae:
					example_song = y_samples[cur_len:cur_len + num_measures]
					cur_notes = example_song * 255
					x = enc_model.predict(np.expand_dims(example_song, 0), batch_size=1)[0]
					cur_len += y_lengths[rand_ix]
					rand_ix += 1
				else:
					rand_ix = np.array([rand_ix], dtype=np.int64)
					x = enc_model.predict(rand_ix, batch_size=1)[0]
					rand_ix = (rand_ix + 1) % model.layers[0].input_dim
				
				if use_pca:
					cur_params = np.dot(x - means, evecs.T) / evals
				else:
					cur_params = (x - means) / stds

				needs_update = True
				audio_reset = True
			if event.key == pygame.K_g:
				audio_pause = True
				audio_reset = True
				midi.samples_to_midi(cur_notes, 'live.mid', 16, note_thresh)
				save_audio = ''
				while True:
					save_audio += audio_callback(None, 1024, None, None)[0]
					if audio_time == 0:
						break
				wave_output = wave.open('live.wav', 'w')
				wave_output.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
				wave_output.writeframes(save_audio)
				wave_output.close()
				audio_pause = False
			if event.key == pygame.K_ESCAPE:
				running = False
				break
			if event.key == pygame.K_SPACE:
				audio_pause = not audio_pause
			if event.key == pygame.K_TAB:
				audio_reset = True
			if event.key == pygame.K_1:
				instrument = 0
			if event.key == pygame.K_2:
				instrument = 1
			if event.key == pygame.K_3:
				instrument = 2
			if event.key == pygame.K_4:
				instrument = 3
			if event.key == pygame.K_c:
				y = np.expand_dims(np.where(cur_notes > note_thresh, 1, 0), 0)
				x = enc_model.predict(y)[0]
				if use_pca:
					cur_params = np.dot(x - means, evecs.T) / evals
				else:
					cur_params = (x - means) / stds
				needs_update = True

	#Check if we need an update
	if needs_update:
		if use_pca:
			x = means + np.dot(cur_params * evals, evecs)
		else:
			x = means + stds * cur_params
		x = np.expand_dims(x, axis=0)
		y = enc([x, 0])[0][0]
		cur_notes = (y * 255.0).astype(np.uint8)
		needs_update = False
	
	#Draw to the screen
	screen.fill(background_color)
	draw_notes()
	draw_sliders()
	draw_controls()
	
	#Flip the screen buffer
	pygame.display.flip()
	pygame.time.wait(10)

#Close the audio stream
audio_stream.stop_stream()
audio_stream.close()
audio.terminate()

