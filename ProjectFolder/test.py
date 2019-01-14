'''import numpy as np
from ProjectFolder.Code.evaluate import pitch_confusion, final_score

x = np.random.binomial(1, 0.005, (300, 500))
y = np.random.binomial(1, 0.003, (300, 500))

pitch_confusion(x, y, vtype='joint', save_path='test.png', description="3")'''


import sys
import argparse
import numpy as np, matplotlib.pyplot as plt
import pretty_midi
import librosa, librosa.display
import itertools
from IPython.display import Audio,display

#parse midi to pretty midi object
midi_fn = '../Audiodaten/AkPnBcht/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.mid'
# SW==>changed: sr = 22050
sr = 16000
pretty_midi.pretty_midi.MAX_TICK = 1e10
pm = pretty_midi.PrettyMIDI(midi_fn)

#synthesize to audio, listen, and compute, view spectrogram
# Package required pip install pyfluidsynth
y = pm.fluidsynth(fs=sr)[:sr*5]
display(Audio(y,rate=sr))
D = librosa.stft(y)
librosa.display.specshow(librosa.amplitude_to_db(D,
                                                 ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()