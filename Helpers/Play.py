import subprocess

from scipy import signal # needed in resampling

import soundfile as sf	# needed in resampling needs PySound(pip install pysound)tested using python3.6.5
import numpy as np

import pyglet # required in playw1
import pygame # required in playw2
import playsound # required in playw3

"""
input : file name, must be a wav file (*.wav)
output : none, needs KeyBpardInterrupt to terminate
needs : pyglet package
Todo : 
various playn with varying output loudness ???
get rid of thread hanging at end
"""
def playw1(f):
    musi = pyglet.media.load(f, streaming = False)
    musi.play()
    pyglet.app.run()  # need ctrl-C to break 


"""
input : file name, must be a wav file (*.wav)
output : none
needs : pygame package
Todo : 
"""
def playw2(f):
    pygame.mixer.init()
    pygame.mixer.music.load(f)
    pygame.mixer.music.play()


"""
input : file name, must be a wav file (*.wav)
output : none
needs : playsound package
Todo : module gi is missing in _playSoundNix
"""
#from playsound import playsound
#def playw3(f):
#    playsound.playsound(f)
#    or playsound(f, block=False)

"""
input : file name, must be a wav file (*.wav)
output : none
needs : . 
Todo : 
"""
def playw4(f):
    subprocess.Popen(['aplay', '-q', f])

"""
input : file name, must be a wav file (*.wav)
output : none
needs : . 
Todo : 
"""
def playw5(f):
    subprocess.call(['vlc', '--play-and-exit', f])


"""
input : file name, must be a mid file (*.mid)
output : none
needs : . 
Todo : 
"""
def playm1(f):
    # todo 
    return




