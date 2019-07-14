# Composer
Generate tunable music using neural networks.

Repository to ["Generating Songs With Neural Networks (Neural Composer)"](https://youtu.be/UWxfnNXlVy8).

## How to install

* Install dependencies in python3 by running `pip install -r requirements.txt`.

## How to run

* Find some dataset to train on. More info on where to find datasets can be found in data/raw/README.md. A default bach dataset is included in the repository.
* Run preprocess_songs.py. This will load all midi files from your midi files folder data/raw/ into data/interim/samples.npy & lengths.npy.
  You can point the script to a location using the --data_folder flag one or multiple times to make it include more paths to  index midi files.
* Run train.py. This will train your network and store the progress from time to time (EPOCHS_TO_SAVE to change storing frequency).
  Only stops if you interrupt it or after 2000 epochs.
* Run composer.py --model_path "e300" where "e300" indicates the folder the stored model is in. Look into the results/history folder to find your trained model folders.

## Composer

The Composer script will load your model trained on the midi files you provided
into the music generator.
Internally, the model tries to compress a song into only 120 numbers called the latent space.
Seen from the other perspective, the latent space numbers encode each song that the model
is trained on. Changing the latent numbers allows us to sample new, unknown songs.
To make the sampling more efficient, we use PCA, a way to extract the dimensions in which there is 
most variations across all songs, to make the influence on the new songs strong. 
The sliders in the composer adjust those numbers and are ordered from most
important/influential (left) to least important/influential (right).  Just the
top 40 are shown on screen.  For more details about how
this works and other fun projects, please check out my
YouTube channel 'CodeParade'.  Have fun making music!

=========================================================
```
Composer Controls:
 Right Click - Reset all sliders
         'R' - Random Song
         'T' - Randomize off-screen sliders
         'M' - Save song as .mid file
         'W' - Save song as .wav file
         'S' - Save slider values to text file
     'Space' - Play/Pause
       'Tab' - Seek to start of song
         '1' - Square wave instrument
         '2' - Sawtooth wave instrument
         '3' - Triangle wave instrument
         '4' - Sine wave instrument
```

## Pretrained models

Below are some pretrained models for different datasets. To use the model, download the zip from the
link below and extract it into the results/history folder of your project. The zip should contain a folder 
named e and some number, indicating the number of epochs the model was trained and some model.h5.
Pay attention when extracting the model not to override one of your own trained, most beloved models!

* Bach dataset: https://drive.google.com/open?id=1P_hOF0v55m1Snzvri5OBVP5yy099Lzgx
