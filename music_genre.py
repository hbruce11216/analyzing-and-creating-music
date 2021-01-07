#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 08:48:03 2020

@author: holdenbruce
"""


# https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8

# https://www.geeksforgeeks.org/how-to-install-librosa-library-in-python/
# ^ how to install 

# Generate a sound / loading an audio file
import librosa
audio_path = 'baby.wav'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))
# <class 'numpy.ndarray'> <class 'int'>
print(x.shape, sr)
# (9029475,) 22050

# This returns an audio time series as a numpy array with a default sampling rate(sr) of 22KHZ mono. We can change this behaviour by saying:
librosa.load(audio_path, sr=44100) #to resample at 44.1KHz, or
librosa.load(audio_path, sr=None) #to disable resampling.
#The sample rate is the number of samples of audio carried per second, measured in Hz or kHz.


### Playing Audio, using IPython.display.Audio
import IPython.display as ipd
ipd.Audio(audio_path)

#### doesn't work with IPython in Spyder :( 






