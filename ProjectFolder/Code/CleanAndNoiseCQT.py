import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

import acoustics
from acoustics.generator import white

import soundfile as sf

import librosa
from librosa import display

fname = "../MAPS_MUS-mond_1_SptkBGAm.wav"
data, fs = sf.read (fname)

laenge = 500000 
links = data[0:laenge,0]
rechts = data[0:laenge,1]

maxdata = np.max(links)
mindata = np.min(links)

ampl = 0.02

wh = ampl * white(laenge)

y, sr = librosa.load(fname)
#print (y.shape) 	# (8036879,)
c = np.abs(librosa.cqt(y[0:laenge], sr=sr))
#print (c.shape)   # (84, 977)  using y[0:laenge]

plt.figure(figsize = (1,2))

plt.subplot(121)
librosa.display.specshow(librosa.amplitude_to_db(c, ref=np.max), sr=sr, y_axis = 'linear')
plt.colorbar() #plt.colorbar(format='%+2.0f dB')
plt.xlabel("time")
plt.ylabel("Frequency [Hz]")
plt.title("CQT of clean wave")

plt.subplot(122)
d = np.abs(librosa.cqt(y[0:laenge] + wh[0:laenge], sr=sr))
librosa.display.specshow(librosa.amplitude_to_db(d, ref=np.max), sr=sr, y_axis = 'linear')
plt.colorbar()
plt.xlabel("time")
plt.ylabel("Frequency [Hz]")
plt.title("CQT of noised wave")

plt.tight_layout()

plt.show()


