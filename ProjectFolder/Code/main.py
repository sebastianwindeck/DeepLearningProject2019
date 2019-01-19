import matplotlib
import datetime
import inspect
import os
import numpy as np
from evaluate import final_score
from extractfeatures import prepareData
from model import Noiser, AMTNetwork
from visualize import visualize_input

matplotlib.use('Agg')

if __name__ == '__main__':

    proj_root = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), '..')

    # Define a parameter structure args
    args = {  # model parameters:
        'model_name': 'baseline', 'init_lr': 1e-1,
        'lr_decay': 'linear',

        # parameters for audio
        'bin_multiple': 3,
        'residual': 'False',
        'hop_length': 512,
        'sr': 16000,
        'spec_type': 'cqt',
        'min_midi': 37,  # 21 corresponds to A0 (lowest tone on a "normal" piano), 27.5Hz
        'max_midi': 92,  # 108 corresponds to  C8 (highest tone on a "normal" piano), 4.2kHz
        'window_size': 7,  # choose higher value than 5

        # training parameters:
        'train_basemodel': True,
        'epochs_on_clean': 1000,
        'epochs_on_noisy': 50,
        'noise_epochs': 10,
        'min_difficulty_on_noisy': 0.19,
        'max_difficulty_on_noisy': 0.60,

        # noise parameters:
        'noise_type': 'simplistic',
        'noise_frames_per_epoch': 100,
        'noise_initial_level': 0.03,
        'noise_increase_factor': 2.5,
        'noise_decrease_factor': 2,
        'balance_classes': True,

        # directories:
        'proj_root': proj_root,  # - root directory of maps (with substructure as given in maps):
        'wav_dir': os.path.join(proj_root, 'Audiodaten'),
        # - directory to store checkpoint files. All files are stored in this directory:
        'checkpoint_root': os.path.join(proj_root, 'Checkpoints',
                                        'train' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
        'basemodel_root': os.path.join(proj_root, 'Basemodel'),

        'maxFramesPerFile': -1,  # set to -1 to ignore
        'maxFrames': -1  # set to -1 to ignore
    }

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
        print('Created checkpoint root.')

    if os.path.exists(args['basemodel_root']):
        print("WARNING: Basemodel root directory already exists!!")
    else:
        os.mkdir(args['basemodel_root'])
        print('Created basemodel root.')

    inputs, outputs, datapath = prepareData(args)
    print("Inputs have shape: ", inputs.shape)
    print("Outputs have shape: ", outputs.shape)
    print("Total number of notes detected in input set ", np.sum(outputs))
    print("Number of 1s in output: ", sum(sum(outputs == 1)))
    print("Number of 0s in output: ", sum(sum(outputs == 0)))
    print("Size of outputs: ", outputs.size)
    print("=> 1s should be weighted ", sum(sum(outputs == 0)) / sum(sum(outputs == 1)))
    input_level = np.mean(inputs)
    print("Average sound level: ", input_level)
    np.append(input_level, np.min(inputs))
    np.append(input_level, np.max(inputs))
    np.append(input_level, np.percentile(inputs, 25))
    np.append(input_level, np.percentile(inputs, 50))
    np.append(input_level, np.percentile(inputs, 75))
    np.save(os.path.join(args['checkpoint_root'], "input_level"), input_level)
    exit()
    visualize_input(inputs, save_path=os.path.join(args['checkpoint_root'], 'input_heatmap.png'))

    # initialize the amt model, and do an initial training
    at = AMTNetwork(args)

    baseModelPath = os.path.join(args['basemodel_root'], 'basemodel')
    evaluatePath = os.path.join(args['checkpoint_root'], 'diagram')
    if args['train_basemodel']:
        # initial training, with clean data:
        print("training initial basemodel")

        at.compilation(outputs, save_path=os.path.join(args['checkpoint_root'], 'balance_weight.png'))
        at.train(inputs, outputs, args=args, epochs=args['epochs_on_clean'], train_descr='initial')
        at.save(model_path=baseModelPath)
    else:
        print("load existing basemodel")
        # Load Basemodel:
        bm = AMTNetwork(args)
        bm.load(baseModelPath)
        bm.compilation(outputs, save_path=os.path.join(args['checkpoint_root'], 'balance_weight.png'))

        # Noise Model to Train:
        at.load(baseModelPath)
        at.compilation(outputs, save_path=os.path.join(args['checkpoint_root'], 'balance_weight.png'))
    # initialize noiser:
    noise_generator = Noiser(noise_type="gaussian", noise_size=args['input_shape'])

    # Track f1 scores of the basemodel that is not further trained to noise
    print("computing initial basemodel scores")
    basemodel_score = bm.getscores(inputs, outputs)
    print('scores of basemodel', basemodel_score)

    # Save Noise levels
    noise_levels = np.zeros(shape=1)
    noise_level = args['noise_initial_level']

    # Initialize base loss
    idx = np.random.randint(0, inputs.shape[0], args['noise_frames_per_epoch'])
    input_clean = inputs[idx]
    y = outputs[idx]
    res_old = at.evaluate_old(input_clean, y)

    # loop over various noise epochs:
    for noiseEpoch in range(args['noise_epochs']):
        # a.	Generate noise candidate (only) with current noise level [advTrain]
        this_noise = noise_generator.generate(args['noise_frames_per_epoch'])

        # indices of data samples to be noised.
        idx = np.random.randint(0, inputs.shape[0], args['noise_frames_per_epoch'])

        classi_change = 0.00
        while True:
            # Combine noise with clean data (noise and audio)
            noisy_X = inputs[idx] + noise_level * this_noise
            print("current noise level before test", noise_level)
            noisy_Xold = inputs[idx] + noise_levels[noiseEpoch] * this_noise
            print("current old noise level before test", noise_levels[noiseEpoch])
            y = outputs[idx]
            classi_change = at.evaluation(noisy_X, res_old, y)
            print("classifier changed by", classi_change)

            if noise_level > 10e8 or noise_level < 10e-8:
                print("Noise Level is: ", noise_level, " in epoch ", noiseEpoch)
                print("BREAK because of size threshold")
                break

            elif classi_change > args['max_difficulty_on_noisy']:
                # “too hard for AMT” -> decrease noise level
                print("too hard")
                noise_level /= args['noise_decrease_factor']
                print('Current noise level' + str(float(noise_level)) + ' in epoch ' + str(noiseEpoch))
                continue  # Jump to the next cycle

            elif classi_change < args['min_difficulty_on_noisy']:
                # “too easy for AMT” -> increase noise level
                noise_level *= args['noise_increase_factor']
                print("too easy")
                print('Current noise level' + str(float(noise_level)) + ' in epoch ' + str(noiseEpoch))
                continue  # Jump to the next cycle

            else:
                print("Noise Level is: ", noise_level, " in epoch ", noiseEpoch)
                print("Noise Level accepted for training")
                # if we reach this point, the classi_perf is in the defined interval
                # => Exit the while loop and train the amt with the new noisy data
                break

        # appending current noise level before training to numpy array "noise_levels"
        noise_levels = np.append(noise_levels, noise_level)

        # Train with noisy samples (for a given number of epochs, with intermed. Result saved)
        this_noise = noise_generator.generate(inputs.shape[0])
        noisy_inputs = inputs + np.random.uniform(0, noise_level, 1) * this_noise
        print("start training")
        at.train(noisy_inputs, outputs, args=args, epochs=args['epochs_on_noisy'],
                 train_descr='noisy_iter_' + str(noiseEpoch))

        # compute loss for next loop evaluation
        res_old = at.evaluate_old(noisy_X, y)

        bm_pred = bm.getscores(noisy_inputs, outputs)
        print("score of basemodel on noisy data", bm_pred)
        basemodel_score = np.append(basemodel_score, bm_pred)
        np.save(os.path.join(args['checkpoint_root'], "noise_levels"), noise_levels)
        np.save(os.path.join(args['checkpoint_root'], "bm_score"), basemodel_score)

        if noiseEpoch != 0 and ((noiseEpoch & (noiseEpoch - 1)) == 0):
            y_pred = at.transcribe(noisy_inputs)
            print(np.max(y_pred, axis=1))
            y_pred = np.around(y_pred, decimals=0)
            print(y_pred.shape)
            y_true = outputs
            print(y_true.shape)
            final_score(y_pred=y_pred, y_true=y_true, description=str(noiseEpoch))

            # Save np array of noise levels
    np.save(os.path.join(args['checkpoint_root'], "noise_levels"), noise_levels)
    print("all noise levels saved")
    np.save(os.path.join(args['checkpoint_root'], "bm_score"), basemodel_score)
    print("all basemodel scores on noise levels saved")

    # Final evaluation:
    final_score(y_pred=y_pred, y_true=y_true, description='final')
    # pitch_confusion(y_pred=y_pred, y_true=y_true, save_path=evaluatePath, description='final')

    print("DONE.")
