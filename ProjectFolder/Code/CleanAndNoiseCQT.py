import numpy as np
import matplotlib.pyplot as plt
from acoustics.generator import white
import soundfile as sf

import librosa
from librosa import display

fname = "../MAPS_MUS-mond_1_SptkBGAm.wav"
data, fs = sf.read (fname)

laenge = 300000
links = data[0:laenge,0]
rechts = data[0:laenge,1]

maxdata = np.max(links)
mindata = np.min(links)

y, sr = librosa.load(fname)
c = np.abs(librosa.cqt(y[0:laenge], sr=sr))

wh = white(laenge)
plt.figure(figsize = (1,3))

plt.subplot(131)
librosa.display.specshow(librosa.amplitude_to_db(c, ref=np.max), sr=sr, y_axis = 'linear')
plt.colorbar()
plt.xlabel("time")
plt.ylabel("Frequency [Hz]")
plt.title("CQT of Clean Signal")

plt.subplot(132)

d = np.abs(librosa.cqt(y[0:laenge] + 0.02*wh[0:laenge], sr=sr))
librosa.display.specshow(librosa.amplitude_to_db(d, ref=np.max), sr=sr, y_axis = 'linear')
plt.colorbar()
plt.xlabel("time")
plt.ylabel("Frequency [Hz]")
plt.title("CQT of Noisy Signal (Noise Level 0.02)")


plt.subplot(133)
d = np.abs(librosa.cqt(y[0:laenge] + 0.34*wh[0:laenge], sr=sr))
librosa.display.specshow(librosa.amplitude_to_db(d, ref=np.max), sr=sr, y_axis = 'linear')
plt.colorbar()
plt.xlabel("time")
plt.ylabel("Frequency [Hz]")
plt.title("CQT of Noisy Signal (Noise Level 0.34)")

plt.tight_layout()

plt.show()


