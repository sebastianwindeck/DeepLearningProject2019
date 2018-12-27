from mido import MidiFile
import numpy as np

import wave	# wave module in python

from scipy.io import wavfile

"""
input : file name, must be a midi file (*.mid)
output : numpy array of type string, containing all MIDI data

needs : mido package  : pip install mido 
 
Todo : check for valid input file
       partial read 
"""
def readMIDI (f):	
    mid = MidiFile(f)
    ostr = []

    for i, track in enumerate (mid.tracks):  
        ostr.append('Track {}: {}\n'.format(i, track.name))
        for msg in track:
            ostr.append(msg)

    return np.array(ostr)


"""
input : input file name, must be a midi file (*.mid);  output filename optional
output : text file containing all MIDI data
         if not specified, output file name = input file name with ".wav" replaced by "_MIDI.txt"

needs : mido package  : pip install mido 
 
Todo : check for valid input file
       partial read 
"""
def MIDI2file (ifi, ofo=''):	
    mid = MidiFile(ifi)
    ostr = []

    if (ofo == ''):
        ofo = ifi.replace(".mid", "_MIDI.txt")
    sw = open(ofo, "w")
    for i, track in enumerate (mid.tracks):  
        sw.write('Track {}: {}\n'.format(i, track.name))
        for msg in track:
            sw.write(str(msg))
            sw.write('\n')
    sw.close()


"""
input : file name, must be a wav file (*.wav)
output : numpy array , containing all data : 1-D if input is mono, 2-D if input is stereo
needs : wave module in python
Todo : check for valid input file
       read n frames (partial read)
"""
def readWAV(f):
    sr = wave.openfp(f, "r")
    nchannels, sampwidth, framerate, nframes, comptype, compname = sr.getparams()  # for future use, seems not available in scipy.io.wavfile

    fs, data = wavfile.read(f) # could also use sr.readframes(n) :  sr.readframes doesn't seem to deliver left / right channels 

    return data



"""
input : input file name, must be a wav file (*.wav);  output filename optional
output : text file containing all data : 1-D if input is mono, 2-D if input is stereo
         if not specified, output file name = input file name with ".wav" replaced by ".txt"
needs : wave module in python
Todo : check for valid input file
       read n frames (partial read)
not very useful because output file size might be huge
better output into an npy, a pcl (?) or some compressed format
"""
def WAV2file(ifi, ofo=''):
    sr = wave.openfp(ifi, "r")
    nchannels, sampwidth, framerate, nframes, comptype, compname = sr.getparams()  # for future use, seems not available in scipy.io.wavfile

    if (ofo == ''):
        ofo = ifi.replace(".wav", "_np.txt")

    fs, data = wavfile.read(ifi) # could also use sr.readframes(n) :  sr.readframes doesn't seem to deliver left / right channels 

    wf = open(ofo, "w")    # open a text file, python wave module 

    for i in range(len(data)):
        print(data, file=wf)
    return data
