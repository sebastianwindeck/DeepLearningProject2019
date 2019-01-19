from scipy.io import wavfile

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

import acoustics
from acoustics.generator import white

import soundfile as sf

fname = "../MAPS_MUS-mond_1_SptkBGAm.wav"
data, fs = sf.read (fname)

laenge = 500000 
links = data[0:laenge,0]
rechts = data[0:laenge,1]

maxdata = np.max(links)
mindata = np.min(links)

ampl = 0.02 * ( maxdata - mindata )

wh = ampl * white(laenge)

sf.write(fname.replace(".wav", "_wh002.wav"), np.stack((links+wh, rechts+wh), axis=1), 44100)
sf.write(fname.replace(".wav", "_nonoise.wav"), np.stack((links, rechts), axis=1), 44100)

plotstart = 15000
plotend = 33000
figu = plt.figure()
axo = figu.add_subplot(211)
axo.set_title('no noise', fontsize=8)
axo.plot(links[plotstart:plotend], color = 'green')  

axw = figu.add_subplot(212)
axw.set_title('white noise', fontsize=8)
axw.plot((links+wh)[plotstart:plotend], color = 'red')  

plt.tight_layout()
plt.show()




