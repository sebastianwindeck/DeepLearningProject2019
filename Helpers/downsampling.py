from scipy import signal 
import wave

import soundfile as sf	# needs PySound		tested using python3.6.5
# use as	python downsampling.py  MAPS_MUS-chpn-p7_SptkBGCl.wav 
# output filename MAPS_MUS-chpn-p7_SptkBGCl_16k_s.wav 
# TODO : parameterize input and output sample rates

from scipy.io import wavfile

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2: 
    print ("needs a wav file .. exiting ")
    exit()

ifname = sys.argv[1]
#python downsampling.py MAPS_MUS-mond_1_SptkBGAm.wav
#python downsampling.py MAPS_MUS-chpn-p7_SptkBGCl.wav
#python downsampling.py MAPS_MUS-bach_846_SptkBGAm.wav

if (not ( ifname.endswith(".wav"))): 
    print ("invalid wav file, extension must be wav")
    exit()

data, fs = sf.read(ifname)
print(data.shape)

up = 160
down = 441

data0_res = signal.resample_poly(data[:,0], up, down)
data1_res = signal.resample_poly(data[:,1], up, down)

#data_res = np.stack((data0_res, data1_res), axis = 1)
#print(data_res.shape)

ofname = ifname.replace(".wav", "_16k_s.wav")
sf.write(ofname, np.stack((data0_res, data1_res), axis=1), 16000)

#"MAPS_MUS-mond_1_SptkBGAm_16k_s.wav"
#"MAPS_MUS-chpn-p7_SptkBGCl_16k_s.wav"
#"MAPS_MUS-bach_846_SptkBGAm_16k_s.wav"

