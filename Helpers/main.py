import numpy as np
from Helpers.Preprocessing import getCQT, resampling
#import Play
from Helpers.IO import readMIDI, readWAV, MIDI2file
from sklearn.utils.validation import check_array, column_or_1d

#ifile_mid = 'mary.mid'

#Kurze Beschreibung was genau welches Dataset ist
ifile_mid = 'MAPS_MUS-mond_1_SptkBGAm.mid'
ifile_wav = 'MAPS_MUS-mond_1_SptkBGAm.wav'	# longer
#Wieso den gleichen namen geben? Dann wird das überschrieben
ifile_wav = 'MAPS_MUS-chpn-p7_SptkBGCl.wav'  # shorter

#Was macht getCQT??
C = getCQT(ifile_wav)

MIDI2file(ifile_mid)

outstr = readMIDI(ifile_mid)
column_or_1d(outstr)

wavContent = readWAV(ifile_wav)
#check_array(wavContent, dtype = 'str' , ensure_min_samples = 1)
check_array(wavContent, ensure_2d= True , ensure_min_samples = 1)

#IO.WAV2file(ifile_wav)	# not very useful, especially when the file size is big, i.e. why need to write to file

#print("starting playw1")
#Play.playw1(ifile_wav)

#print("starting playw2")
#Play.playw2(ifile_wav)

#print("starting playw3")
#Play.playw3(ifile_wav)

#print("starting playw5")
#Play.playw5(ifile_wav)

resampling("MAPS_MUS-chpn-p7_SptkBGCl.wav", 160, 441)
#resampling.py MAPS_MUS-mond_1_SptkBGAm.wav
#resampling.py MAPS_MUS-chpn-p7_SptkBGCl.wav
#resampling.py MAPS_MUS-bach_846_SptkBGAm.wav
#"MAPS_MUS-mond_1_SptkBGAm_16k_s.wav"
#"MAPS_MUS-chpn-p7_SptkBGCl_16k_s.wav"
#"MAPS_MUS-bach_846_SptkBGAm_16k_s.wav"

print("starting to play the resampled wav")
Play.playw5(ifile_wav.replace(".wav", "_16k_s.wav"))

