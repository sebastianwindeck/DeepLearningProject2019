from scipy.io import wavfile

import sys
import numpy as np

if (not ( sys.argv[1].endswith(".wav"))): 
    print ("invalid wav file, extension must be wav")
    exit()

fs, data = wavfile.read(sys.argv[1])	# muss die Endung wav haben

wf = open(sys.argv[1].replace(".wav", "_WAV.txt"), "w")

for i in range (min(100000, len(data))):
    print(data[i, 0], data[i, 1], file=wf)
