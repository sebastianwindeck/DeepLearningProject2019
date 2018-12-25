from scipy import signal # needed in resampling

import numpy as np

import wave

import soundfile as sf	# requires PySound		tested using python3.6.5

from scipy.io import wavfile


"""
input : file name, must be a mid file (*.mid)
output : none
needs : . 
Todo : parameterize input and output sample rates
get rid of the hard coded 16000 when writing
check for valid sample rates
distinguish between mono and stereo
"""
def resampling(ifi, up=160, down=441):
    data, fs = sf.read(ifi)
    
    data0_res = signal.resample_poly(data[:,0], up, down)
    data1_res = signal.resample_poly(data[:,1], up, down)
    
    ofi = ifi.replace(".wav", "_16k_s.wav")
    sf.write(ofi, np.stack((data0_res, data1_res), axis=1), 16000)
