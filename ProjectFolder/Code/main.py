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

#import foolbox
#from foolbox import Adversarial
#from foolbox.distances import MeanSquaredDistance

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect

# import pretty_midi
import sys
import tensorflow as tf

import sklearn
from sklearn.metrics import precision_recall_fscore_support

from operator import concat


import io  # has function prepareData(args)

import datetime

from ProjectFolder.Code.io import prepareData
from ProjectFolder.Code.model import Noiser, AMTNetwork

if __name__ == '__main__':

    proj_root = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), '..')

    #   TODO:   [all]       i. Init everything, namely: noise level
    # Define a parameter structure args
    args = {# model parameters:
        'model_name': 'baseline',
        #für init lr geht auch 0.1
        'init_lr': 1e-2,
        'lr_decay': 'linear',

        # parameters for audio
        'sr': 16000,
        'spec_type': 'cqt',
        'bin_multiple': 3,
        'residual': 'False',
        'min_midi': 37,  # 21 corresponds to A0 (lowest tone on a "normal" piano), 27.5Hz
        'max_midi': 92,  # 108 corresponds to  C8 (highest tone on a "normal" piano), 4.2kHz
        'window_size': 5,
        'hop_length': 512,

        # training parameters: ==> currently just some random numbers...
        'epochs_on_clean': 100,
        'epochs_on_noisy': 10,
        'noise_epochs': 20,
        'min_difficulty_on_noisy': 0.05,  # just a random value...
        'max_difficulty_on_noisy': 0.15,  # just a random value...

        # noise parameters:
        'noise_type': 'simplistic',
        'noise_frames_per_epoch': 20,  # just a random value...
        'noise_initial_level': 0.001,  # just a random value...
        'noise_increase_factor': 1.5,  # just a random value...
        'noise_decrease_factor': 1.5,  # just a random value...

        # directories:
        'proj_root': proj_root,
        # - root directory of maps (with substructure as given in maps):
        'wav_dir': os.path.join(proj_root, 'Audiodaten'),
        # - directory to store checkpoint files. All files are stored in this directory:
        'checkpoint_root': os.path.join(proj_root, 'Checkpoints', \
                                        'train' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),

        # "quick-test parameters", such that only a few data samples are used
        'maxFramesPerFile': 20,  # set to -1 to ignore
        'maxFrames': 50         # set to -1 to ignore
    }  # Feel free to add more parameters if needed.

    # derived parameters:
    args['note_range'] = args['max_midi'] - args['min_midi'] + 1
    args['feature_bins'] = args['note_range'] * args['bin_multiple']
    args['input_shape'] = (args['window_size'], args['feature_bins'])
    args['input_shape_channels'] = (args['window_size'], args['feature_bins'], 1)
    args['n_bins'] = args['note_range'] * args['bin_multiple']

    if os.path.exists(args['checkpoint_root']):
        print("WARNING: Checkpoint root directory already exists!!")
    else:
        os.mkdir(args['checkpoint_root'])

    #   TODO:   [Andreas]   ii. CQT calculate [advTrain]
    # comment AS: Currently, this calculates the features directly from the .wav files. I see two options:
    # 1. change this, such that the features are calculated on individual snippets of the audio files.
    # 2. leave it as is, and generate/add the noise to the cqt feature files. Use librosa.icqt whenever we want
    #    to "listen" to the noise.
    #    --> [Malte] I would choose 2.
    # initially, we have decided to go for option 1. However, i propose to stay with option 2, for the following
    # reasons:
    # - it's easier
    # - it's already implemented (hence also easier to compare with literature results)
    # - samples of approx. 32ms are anyway to short to listen to.
    # - icqt is not perfect, but should be sufficiently good to describe e.g. the frequency distribution of the noise.
    inputs, outputs, datapath = prepareData(args)

    # initialize the amt model, and do an initial training
    at = AMTNetwork(args)

    train_basemodel = True

    if train_basemodel:

    # at.init_amt()

    # initial training, with clean data:
        at.train( inputs, outputs, args['epochs_on_clean'], train_descr='initial')
    #   TODO:	[Sebastian] iii. Train base model (for a given number of epochs, with intermed.
    #                               Result saved) kerasTrain ->parameter reduzieren]

    # save parameters after initial training
        base_weights = at.get_weights() #numpy array of all weights
        cwd = os.getcwd()

        path = cwd + "base_weights"
        path = os.path.abspath(path)
        np.save(path, base_weights)



    #um parameter aufzurufen
    #model.set_weights()

    if not train_basemodel:
        base_weights = np.load(path)

    #  Aufrufen Base weights

    # initialize noiser:
    noise_generator = Noiser(noise_type="simplistic", noise_size=args['input_shape'])

    # loop over various noise epochs:
    #   DONE:   [all]        v.	For noiseEpochs = 1 … XXX
    for noiseEpoch in range(args['noise_epochs']):
        # a.	Generate noise candidate (only) with current noise level [advTrain]
        this_noise = noise_generator.generate(args['noise_frames_per_epoch'])
        # comment AS: ich schlage vor, dass wir das Noise 1x generieren und dananch mit dem Level skalieren.
        #             wir hatten das anders vorgeschlagen, aber ich denke so ist es besser, weil wir das noise
        #             und die Intensität separat kontrollieren können. Wenn wir neues Noise mit neuer Intensität
        #             generieren wissen wir nachher nicht, ob die Veränderung jetzt auf die neue Noise-instanz,
        #             auf die veränderte Intensität, oder auf beides zusammen zurückzuführen ist.

        # incices of data samples to be noised.
        idx = np.random.randint(0, inputs.shape[0], args['noise_frames_per_epoch'])
        noise_level = args['noise_initial_level']
        while True:
            # b.	Combine noise with clean data (noise and audio)
            noisy_X = inputs[None, idx] + noise_level * this_noise
            # c.	CQT
            # TODO: calculate CQT based on noisy_X. decide whether we work with cqt values of waveforms...

            #  TODO: d.	Evaluate performance of classifier based on noise candidate
            classi_perf = 0.42
            if classi_perf > args['max_difficulty_on_noisy']:
                # “too hard for AMT” -> decrease noise level
                noise_level /= args['noise_decrease_factor']
                continue # Jump to the next cycle

            if classi_perf < args['min_difficulty_on_noisy']:
                # “too easy for AMT” -> increase noise level
                noise_level *= args['noise_increase_factor']
                continue # Jump to the next cycle

            # if we reach this point, the classi_perf is in the defined interval
            # => Exit the while loop and train the amt with the new noisy data
            break

        # TODO: i.	Break -> save Noise audio file

        # Train with noisy samples (for a given number of epochs, with intermed. Result saved)
        # TODO: probably needs some refinements
        at.train( inputs, outputs, args['epochs_on_noisy'], train_descr='noisy_iter'+str(noiseEpoch))

        # TODO:   [Tanos]   3.	Save intermediate results

    # end for noiseEpoch in range(args['noise_epochs'])

#   TODO:   [all]       vi.	Overall eval:
#                           1.	F1 score compared to noise level
#                           2.	Confusion matrix (heat maps, for e.g. 4 noise levels)

print("DONE.")