"""
    funzt nur mit python2.7
"""

from __future__ import print_function
import midi
import sys

if (not ( sys.argv[1].endswith(".mid"))): 
    print ("invalid MIDI file, extension must be mid")
    exit()

pattern = midi.read_midifile(sys.argv[1])	# muss die Endung .mid haben 

mf = open(sys.argv[1].replace(".mid", "_MIDI.txt"), "w")

print (pattern, file = mf)
