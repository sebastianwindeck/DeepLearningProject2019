from scipy import signal # needed in resampling

import numpy as np

import wave

import soundfile as sf	# requires PySound		tested using python3.6.5

from scipy.io import wavfile


import librosa		# required by cqt
from librosa import display


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

"""
Input: filepath to .wav file
Output: CQT
Todo: error checking
"""
def getCQT(f):
    y, sr = librosa.load(f)
#example:  y,sr = librosa.load("MAPS_MUS-chpn-p7_SptkBGCl_16k_s.wav")

    return np.abs(librosa.cqt(y, sr=sr))


"""
Input: a string
Output: bool, True if input string is a valid representation of a float, else False
"""
def is_valid(s):
     try:
         float(s)
         return True 
     except ValueError:
         return False 


"""
Input: a string representing a file name, e.g. "MAPS_MUS-mond_1_SptkBGAm.txt"
       r sampling rate
Output: a 2 dimensional numpy array, representing a piano roll, 0 = key released, 1 = key pressed
        the number of rows = 109, representing the MidiPitch; the 88 keys of a piano : 21 to 109, middle C = code 60
        the number of columns representing the length of the piece = seconds(length of piece) * samplerate
Todo: optimize for speed; get rid of hard coded cinstants
"""
def createRoll(f, r):
    roll_len = 0
    #there are a number of ways to find the length of the piece, the following is not the best way but simple
    # has to scan through the whole file because the last line might not contain the highest OffsetTime
    # one other way is to read in the wav file to find the length of the piece
    # for the pieces in MAPS , the *.txt files are not so long, hence reading in twice is not bad
    with open(f, "rb") as ifi:
         for zeile in ifi:
             tokens = zeile.split()
             if ( is_valid(tokens[0]) and is_valid(tokens[1])  and tokens[2].isdigit()): 
                 ende = np.int(np.ceil(np.float(tokens[1] ) * r))
                 if (ende > roll_len): roll_len = ende
    
    roll = np.zeros(([109, roll_len ]))    # initialized to 0 = key not pressed
    
    with open(f, "rb") as ifi:
         for zeile in ifi:
             tokens = zeile.split()
             if (is_valid(tokens[0]) and is_valid(tokens[1])  and tokens[2].isdigit()): 
                 anfang = np.int(np.ceil(np.float(tokens[0]) * r))
                 ende = np.int(np.ceil(np.float(tokens[1] ) * r))
                 note = np.int(tokens[2])
                 roll [note, anfang:(ende+1) ] = 1
    
    #for testing
    #for zi in range(109):
    #    if (np.sum(roll[zi,:] > 0)):
    #        print(zi, roll[zi, :])
    return roll
    

"""
Input: a 2-dimensional vector 
       n: length of chunks
Output: a 3-dimensional vector obtained from the input columns divided into chunks of length n  
e.g.  x = np.array(([  
                       [1, 2, 3, 4, 5, 6, 7, 8], 
                       [201, 202, 203, 204, 205, 206, 207, 208]
                    ])) 
      y = breakn(x, 4)
        -> y    = [[ [1, 2, 3, 4], 
                     [201, 202, 203, 204]
                   ], 
                   [ [5, 6, 7, 8], 
                     [205, 206, 207, 208]
                   ]
                  ]
ToDo: get rid of loop
"""
def breakn(x, n):
    n0 = len(x[0,:])
    d0 = x.shape[0]
    d1 = x.shape[1]

    # need to pad with 0 up to length divisible by n, i.e. add columns 
    #extra_col =  n - n0 % n
    #print("extra_col:", extra_col)
    #xp = np.pad(x, pad_width = ((0,0), (0, extra_col)), mode ='constant', constant_values = (0,0))   # takes too long
    #d1n = xp.shape[1]
    #d0n = int(d1n / n)

    d0n = int(n0 / n)

    xn = np.zeros(([d0n, d0, n]))

    for i in range(d0n):
        xn[i, :, :] = x[0:d0, (i*n):(i*n+n)]
    return xn
    
"""
Input: x : numpy array 
Output: (x - mean(x))/stdev(x)
Todo: error checking
"""
def scale(x):
    return (x - np.mean(x)) / np.std(x)
