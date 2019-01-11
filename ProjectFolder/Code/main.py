import datetime
import inspect
import os

import numpy as np

from evaluate import pitch_confusion, final_score
from extractfeatures import prepareData
from model import Noiser, AMTNetwork

# import foolbox
# from foolbox import Adversarial
# from foolbox.distances import MeanSquaredDistance
# import pretty_midi

if __name__ == '__main__':

    proj_root = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), '..')

    # Define a parameter structure args
    args = {  # model parameters:
        'model_name': 'baseline',  # für init lr geht auch 0.1
        'init_lr': 1e-2, 'lr_decay': 'linear',

        # parameters for audio
        'sr': 16000, 'spec_type': 'cqt', 'bin_multiple': 3, 'residual': 'False', 'min_midi': 21,
        # 21 corresponds to A0 (lowest tone on a "normal" piano), 27.5Hz
        'max_midi': 108,  # 108 corresponds to  C8 (highest tone on a "normal" piano), 4.2kHz
        'window_size': 7,  # choose higher value than 5
        'hop_length': 512,

        # training parameters: ==> currently just some random numbers...
        'epochs_on_clean': 100, 'epochs_on_noisy': 10, 'noise_epochs': 20, 'min_difficulty_on_noisy': 0.05,
        # just a random value...
        'max_difficulty_on_noisy': 0.15,  # just a random value...

        # noise parameters:
        'noise_type': 'simplistic', 'noise_frames_per_epoch': 20,  # just a random value...
        'noise_initial_level': 0.001,  # just a random value...
        'noise_increase_factor': 1.5,  # just a random value...
        'noise_decrease_factor': 1.5,  # just a random value...

        # directories:
        'proj_root': proj_root,  # - root directory of maps (with substructure as given in maps):
        'wav_dir': os.path.join(proj_root, 'Audiodaten'),
        # - directory to store checkpoint files. All files are stored in this directory:
        'checkpoint_root': os.path.join(proj_root, 'Checkpoints', \
                                        'train' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
        'basemodel_root': os.path.join(proj_root, 'Basemodel'),
        # "quick-test parameters", such that only a few data samples are used
        'maxFramesPerFile': 20,  # set to -1 to ignore
        'maxFrames': 50  # set to -1 to ignore
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

    if os.path.exists(args['basemodel_root']):
        print("WARNING: Basemodel root directory already exists!!")
    else:
        os.mkdir(args['basemodel_root'])

    # comment AS: Currently, this calculates the features directly from the .wav files. I see two options:
    # 1. change this, such that the features are calculated on individual snippets of the audio files.
    # 2. leave it as is, and generate/add the noise to the cqt feature files. Use librosa.icqt whenever we want
    #    to "listen" to the noise.
    #    --> [Malte] I would choose 2.
    #    --> [Sebastian] I would also choose 2.
    # initially, we have decided to go for option 1. However, i propose to stay with option 2, for the following
    # reasons:
    # - it's easier
    # - it's already implemented (hence also easier to compare with literature results)
    # - samples of approx. 32ms are anyway to short to listen to.
    # - icqt is not perfect, but should be sufficiently good to describe e.g. the frequency distribution of the noise.
    # => ANGENOMMEN
    inputs, outputs, datapath = prepareData(args)

    # initialize the amt model, and do an initial training
    at = AMTNetwork(args)

    train_basemodel = True
    baseModelPath = os.path.join(args['basemodel_root'], 'basemodel')
    evaluatePath = os.path.join(args['checkpoint_root'], 'diagram')
    if train_basemodel:
        # initial training, with clean data:
        at.train(inputs, outputs, epochs=args['epochs_on_clean'], train_descr='initial')

        at.save(baseModelPath)

    else:
        at.load(baseModelPath)

    # initialize noiser:
    noise_generator = Noiser(noise_type="simplistic", noise_size=args['input_shape'])

    noise_levels = np.zeros(shape=1)
    noise_level = args['noise_initial_level']

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
        # => ANGENOMMEN

        # indices of data samples to be noised.
        idx = np.random.randint(0, inputs.shape[0], args['noise_frames_per_epoch'])

        classi_change = 0.00
        while True:
            # b.	Combine noise with clean data (noise and audio)
            noisy_X = inputs[idx] + noise_level * this_noise
            noisy_Xold = inputs[idx] + noise_levels[noiseEpoch] * this_noise
            y = outputs[idx]
            classi_change = at.evaluation(noisy_X, noisy_Xold, y)

            #  TODO: d.	Evaluate performance of classifier based on noise candidate
            #  TODO: [Malte] funktion besteht, muss hier aufgerufen werden. Intervall des gesuchten Noise-Veränderung festlegen durch Probieren.
            #

            if classi_change > args['max_difficulty_on_noisy']:
                # “too hard for AMT” -> decrease noise level
                noise_level /= args['noise_decrease_factor']
                continue  # Jump to the next cycle

            if classi_change < args['min_difficulty_on_noisy']:
                # “too easy for AMT” -> increase noise level
                noise_level *= args['noise_increase_factor']
                continue  # Jump to the next cycle

            # if we reach this point, the classi_perf is in the defined interval
            # => Exit the while loop and train the amt with the new noisy data
            break
        # appending current noise level before training to numpy array "noise_levels"
        noise_levels = np.append(noise_levels, noise_level)

        # Train with noisy samples (for a given number of epochs, with intermed. Result saved)
        # TODO: probably needs some refinements => look ok for now.
        this_noise = noise_generator.generate(inputs.shape[0])
        noisy_inputs = inputs + np.random.uniform(0, noise_level, 1) * this_noise
        at.train(noisy_inputs, outputs, args['epochs_on_noisy'], train_descr='noisy_iter_' + str(noiseEpoch))

        if noiseEpoch != 0 and ((noiseEpoch & (noiseEpoch - 1)) == 0):
            y_pred = at.transcribe(noisy_inputs)
            print(np.max(y_pred, axis=1))
            y_pred = np.around(y_pred,decimals=0)
            print(y_pred.shape)
            y_true = outputs
            print(y_true.shape)
            final_score(y_pred=y_pred, y_true=y_true, description=str(noiseEpoch))
            pitch_confusion(y_pred=y_pred, y_true=y_true, save_path=evaluatePath, description=str(noiseEpoch))

    # Save np array of noise levels
    np.save("noise_levels", noise_levels)

    # end for noiseEpoch in range(args['noise_epochs'])

    # Final evaluation:
    #                           1.	F1 score compared to noise level
    #                           2.	Confusion matrix (heat maps, for e.g. 4 noise levels)
    final_score(y_pred=y_pred, y_true=y_true, description='final')
    pitch_confusion(y_pred=y_pred, y_true=y_true, save_path=evaluatePath, description='final')

    print("DONE.")
