# -*- coding: utf-8 -*-
"""
Starting with adversarial training.
Jan  2 2019

@author: andre
"""

from __future__ import print_function, division

from keras import metrics
from keras import backend as K

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, add

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model, load_model

from keras.optimizers import Adam, SGD

from keras.utils import plot_model

import foolbox
from foolbox import Adversarial
from foolbox.distances import MeanSquaredDistance


import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pretty_midi
import sys
import tensorflow as tf

import sklearn
from sklearn.metrics import precision_recall_fscore_support

from config import create_config, load_config
from localsearch import LocalSearchAttack



"""
Handling Audio data
mostly taken directly from preprocess.py, data handling changed such that the 
data from all files is in one array
"""
data_dir = '../maps/'

sr = 22050
hop_length = 512
window_size = 7
# changed_AS:
#min_midi = 21
#max_midi = 108
min_midi = 21+8+8
max_midi = 108-8-8
note_range = max_midi - min_midi + 1

def wav2inputnp(audio_fn,spec_type='cqt',bin_multiple=3):
    print("wav2inputnp")
    bins_per_octave = 12 * bin_multiple #should be a multiple of 12
    # change_AS
    # n_bins = (max_midi - min_midi + 1) * bin_multiple
    n_bins = note_range * bin_multiple

    #down-sample,mono-channel
    y,_ = librosa.load(audio_fn,sr)
    S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                      bins_per_octave=bins_per_octave, n_bins=n_bins)
    S = S.T

    #S = librosa.amplitude_to_db(S)
    S = np.abs(S)
    minDB = np.min(S)
    print(np.min(S),np.max(S),np.mean(S))
    S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)

    windows = []

    # IMPORTANT NOTE:
    # Since we pad the the spectrogram frame,
    # the onset frames are actually `offset` frames.
    # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
    # starting at frame 0 of the padded spectrogram
    for i in range(S.shape[0]-window_size+1):
        w = S[i:i+window_size,:]
        windows.append(w)

    #print inputs
    x = np.array(windows)
    return x


def mid2outputnp(pm_mid,times):
    piano_roll = pm_mid.get_piano_roll(fs=sr,times=times)[min_midi:max_midi+1].T
    piano_roll[piano_roll > 0] = 1
    return piano_roll


def joinAndCreate(basePath,new):
    newPath = os.path.join(basePath,new)
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    return newPath


def prepareData(args):
    #params
    print("Starting preprocessing")
    path = os.path.join('models',args['model_name'])
    #config = load_config(os.path.join(path,'config.json'))

    bin_multiple = int(args['bin_multiple'])
    spec_type = args['spec_type']

    framecnt = 0
    # maxFramesPerFile = 100 # set to -1 to ignore 
    maxFrames = 100 # set to -1 to ignore 
    
    # hack to deal with high PPQ from MAPS
    # https://github.com/craffel/pretty-midi/issues/112
    pretty_midi.pretty_midi.MAX_TICK = 1e10

    inputs,outputs = [],[]
    addCnt, errCnt = 0,0

    for s in os.listdir(data_dir):
        subdir = os.path.join(data_dir,s)
        if not os.path.isdir(subdir):
            continue
        # recursively search in subdir
        print(subdir)
        for dp, dn, filenames in os.walk(subdir):
            # in each level of the directory, look at filenames ending with .mid
            for f in filenames:
                # if there exists a .wav file and .midi file with the same name

                if f.endswith('.wav'):
                    audio_fn = f
                    fprefix = audio_fn.split('.wav')[0]
                    mid_fn = fprefix + '.mid'
                    txt_fn = fprefix + '.txt'
                    print("Handling files {}".format(fprefix))
                    if mid_fn in filenames:
                        # wav2inputnp
                        audio_fn = os.path.join(dp,audio_fn)
                        # mid2outputnp
                        mid_fn = os.path.join(dp,mid_fn)

                        pm_mid = pretty_midi.PrettyMIDI(mid_fn)

                        inputnp = wav2inputnp(audio_fn,spec_type=spec_type,bin_multiple=bin_multiple)
                        times = librosa.frames_to_time(np.arange(inputnp.shape[0]),sr=sr,hop_length=hop_length)
                        outputnp = mid2outputnp(pm_mid,times)

                        # check that num onsets is equal
                        if inputnp.shape[0] == outputnp.shape[0]:
                            print("adding to dataset fprefix {}".format(fprefix))
                            addCnt += 1
                            framecnt += inputnp.shape[0]
                            print("framecnt is {}".format(framecnt))
                            inputs.append(inputnp)
                            outputs.append(outputnp)
                        else:
                            print("error for fprefix {}".format(fprefix))
                            errCnt += 1
                            print(inputnp.shape)
                            print(outputnp.shape)
                            
                if maxFrames>0 and framecnt>maxFrames:
                    print("have enought frames, leaving {}".format(subdir))
                    break
            if maxFrames>0 and framecnt>maxFrames:
                print("have enought frames, leaving {}".format(data_dir))
                break

        if maxFrames>0 and framecnt>maxFrames:
            print("have enought frames, leaving {}".format(data_dir))
            break

        print("{} examples in dataset".format(addCnt))
        print("{} examples couldnt be processed".format(errCnt))
    

    # concatenate dynamic list to numpy list of example
    if addCnt:
        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        fn = subdir.split('/')[-1]
        if not fn:
            fn = subdir.split('/')[-2]
        #fn += '.h5'
        # save inputs,outputs to hdf5 file
        datapath = joinAndCreate(path,'data')
        #fnpath = joinAndCreate(datapath,fn)

        mmi = np.memmap(filename=os.path.join(datapath,'input2.dat'), 
                        mode='w+',shape=inputs.shape)
        mmi[:] = inputs[:]
        mmo = np.memmap(filename=os.path.join(datapath,'output2.dat'), 
                        mode='w+',shape=outputs.shape)
        mmo[:] = outputs[:]
        del mmi
        del mmo
    
    return inputs, outputs, path

# End audio stuff

""" 
helper to read data files
"""
def readmm(d,args):
    ipath = os.path.join(d,'input.dat')
    # changed_AS
    # note_range = 88
    # note_range = int(args['max_midi'])-int(args['min_midi'])+1
    n_bins = int(args['bin_multiple']) * note_range
    # n_bins = self.n_bins
    # note_range = self.note_range
    
    window_size = 7
    mmi = np.memmap(ipath, mode='r')
    i = np.reshape(mmi,(-1,window_size,n_bins))
    opath = os.path.join(d,'output.dat')
    mmo = np.memmap(opath, mode='r')
    o = np.reshape(mmo,(-1,note_range))
    return i,o


class AdvTrain():
    def __init__(self):
        np.random.seed(42)
        
        args = {'model_name': 'baseline',
                'spec_type': 'cqt',
                'init_lr': '1e-2',
                'lr_decay': 'linear',
                'bin_multiple': '3',
                'residual': 'False',
                'min_midi': '37',
                'max_midi': '92'}

        self.min_midi = int(args['min_midi'])
        self.max_midi = int(args['max_midi'])
        self.bin_multiple = int(args['bin_multiple'])
        self.note_range = int(args['max_midi'])-int(args['min_midi'])+1
        self.n_bins = int(args['bin_multiple']) * note_range
        self.init_lr = float(args['init_lr'])
        self.args = args
        
        self.sr = 22050
        self.hop_length = 512
        self.window_size = 7
        self.feature_bins = self.note_range * self.bin_multiple
        self.input_shape = (self.window_size, self.feature_bins)
        self.input_shape_channels = (self.window_size, self.feature_bins,1)
        
        self.inputs, self.outputs, self.path = prepareData(args)
    
        self.latent_dim = 100

        #optimizer = Adam(0.0002, 0.5)

        # initialize the AMT network:
        self.amt_net = self.init_amt()
        
        # initialize the noise generation
        # init_noiser(self):
        self.criterion = foolbox.criteria.Misclassification()
            

    """
    initialize the convolutional neural network for automatic music 
    transcription (AMT). The goal is to get as many correct transcriptions as
    possible given the (potentially noisy) music. For the time being, this is 
    just a copy of baseline_model(). 
    TODO: verify / optimize network architecture
    """
    def init_amt(self): 
        inputs = Input(shape = self.input_shape)
        reshape = Reshape(self.input_shape_channels)(inputs)
    
        #normal convnet layer (have to do one initially to get 64 channels)
        conv1 = Conv2D(50,(5,25),activation='tanh')(reshape)
        do1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(1,3))(do1)
    
        conv2 = Conv2D(50,(3,5),activation='tanh')(pool1)
        do2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(1,3))(do2)
    
        flattened = Flatten()(pool2)
        # changed_AS
        # fc1 = Dense(1000, activation='sigmoid')(flattened)
        fc1 = Dense(100, activation='sigmoid')(flattened)
        do3 = Dropout(0.5)(fc1)
    
        # changed_AS
        # fc2 = Dense(200, activation='sigmoid')(do3)
        fc2 = Dense(50, activation='sigmoid')(do3)
        do4 = Dropout(0.5)(fc2)
        outputs = Dense(note_range, activation='sigmoid')(do4)
    
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', 
                      optimizer=SGD(lr=self.init_lr,momentum=0.9))
        model.summary()
        plot_model(model, to_file=os.path.join(self.path, 'model.png'))

        self.model_ckpt = os.path.join(self.path, 'ckpt.h5')
        return model
    
  
    def train_amt(self, myData, myLabels, prct_val = 0.2):
        # n_data = myData.shape[0]
        # n_data_val = round(prct_val*n_data)
        # idx = np.random.randint(0, n_data, n_data_val)
        
        # data_val = myData[idx]
        # labl_val = myLabels[idx]
        # data_train = myData[-idx]
        # labl_train = myLabels[-idx]
        
        myLoss = self.amt_net.train_on_batch(myData, myLabels)
        return myLoss
    
        
    """ 
    initialize the noise generator that generates noise that should disturb the 
    AMT network. The goal is to minimize the noise level while getting a wrong
    result from the AMT network. 
    Currently using just a "salt-and-pepper" attack taken from the adversarial
    vision challenge for NIPS 2018 taken from github:
        https://gitlab.crowdai.org/adversarial-vision-challenge/saltnpepper_untargeted_attack_baseline/tree/master
        
    TODO: What is a good measure for the noise level?
    TODO: how to make this smarter?
    """
    def init_noiser(self):
        # def run_attack(model, image, label):
        #criterion = foolbox.criteria.Misclassification()
        #attack = foolbox.attacks.SaltAndPepperNoiseAttack(model, criterion)
        #return attack(image, label, epsilons=50, repetitions=10)
        pass


    def run_attack(self, X_train, Y_train):
        
        # create an adversarial example
        # choose a random sample:
        idx = np.random.randint(0, X_train.shape[0], 1)
        
        myAdvSample = Adversarial(self.amt_net, self.criterion,
                                  X_train[idx], Y_train[idx], 
                                  MeanSquaredDistance(bounds = [0, X_train.max()]) )
        
        myAttack = LocalSearchAttack(myAdvSample, unpack=False,
                 r=1.5, p=10., d=5, t=5, R=150)
        
        # get best adversarial found
        return myAttack, Y_train[idx]
    
        """
        A black-box attack based on the idea of greedy local search.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.
        r : float
            Perturbation parameter that controls the cyclic perturbation;
            must be in [0, 2]
        p : float
            Perturbation parameter that controls the pixel sensitivity
            estimation
        d : int
            The half side length of the neighborhood square
        t : int
            The number of pixels perturbed at each round
        R : int
            An upper bound on the number of iterations
        """            
    
    def main_loop(self, maxIter = 10):
        # prepare the data
        # define hold-out set:
        n_data = self.inputs.shape[0]
        n_data_hold_out = round( 0.2 * n_data )
        idx = np.random.randint(0, n_data, n_data_hold_out)
        
        X_train = self.inputs[idx]
        Y_train = self.outputs[idx]


        # initial training for amt
        newLoss = self.train_amt(X_train, Y_train)
        print ("Initial training loss: %f" % (newLoss))
        
        for epoch in range(maxIter):
            print("** Starting iteration %d" % (epoch))
            
            # train noiser based on current model
            attack, true_Y = self.run_attack(X_train, Y_train)
            # append this to the current data
            # TODO: maybe just replace? Or sample afterwards to run training?
            X_train.append(attack.image())
            Y_train.append(true_Y)
            print("Found noise with intensity %.2f for label %d" % (attack.distance, true_Y))
            
            # retrain model based on obtained noise pattern
            newLoss = self.train_amt(self, X_train, Y_train)
            print ("Training loss: %f, acc.: %.2f%%" % (newLoss[0], 100*newLoss[1]))
                       
        
if __name__ == '__main__':
    at = AdvTrain()
    at.main_loop()
