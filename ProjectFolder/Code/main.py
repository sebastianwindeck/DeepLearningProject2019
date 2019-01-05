
import argparse

import io # has function prepareData(args)
import model # has class AMTNetwork
import time

if __name__ == '__main__':

#   TODO:   [all]       i. Init everything, namely: noise level
        # Define a parameter structure args
        args = {# model parameters:
                'model_name': 'baseline',
                'spec_type': 'cqt',
                'init_lr': 1e-2,
                'lr_decay': 'linear',
                'bin_multiple': 3,
                'residual': 'False',
                'min_midi': 37, # 21 corresponds to A0 (lowest tone on a "normal" piano), 27.5Hz
                'max_midi': 92, # 108 corresponds to  C8 (highest tone on a "normal" piano), 4.2kHz
                'window_size': 7,

                # training parameters: ==> currently just some random numbers...
                'epochs_on_clean': 100,
                'epochs_on_noisy': 10,
                'noise_epochs': 20,
                'min_difficulty_on_noisy': 0.05, # just a random value...
                'max_difficulty_on_noisy': 0.15, # just a random value...

                # noise parameters:
                'noise_type': 'simplistic',
                'noise_frames_per_epoch': 20, # just a random value...
                'noise_initial_level': 0.001, # just a random value...
                'noise_increase_factor': 1.5, # just a random value...
                'noise_decrease_factor': 1.5, # just a random value...

                # directories:
                'wav_dir': '../maps/', # root directory of maps (with substructure as given in maps)
                'checkpoint_root': '../Checkpoints/train' + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

                # "quick-test parameters", such that only a few data samples are used
                'maxFramesPerFile': 200,  # set to -1 to ignore
                'maxFrames': 5000         # set to -1 to ignore
        } # Feel free to add more parameters if needed.

        #   TODO:   [Andreas]   ii. CQT calculate [advTrain]
        # comment AS: Currently, this calculates the features directly from the .wav files. I see two options:
        # 1. change this, such that the features are calculated on individual snippets of the audio files.
        # 2. leave it as is, and generate/add the noise to the cqt feature files. Use librosa.icqt whenever we want
        #    to "listen" to the noise.
        inputs, outputs, path = prepareData(args)

        # initialize the amt model, and do an initial training
        at = AMTNetwork(args)
        at.init_amt()

        # initial training, with clean data:
        at.train( inputs, outputs, args['epochs_on_clean'], train_descr='initial')
        #   TODO:	[Sebastian] iii. Train base model (for a given number of epochs, with intermed.
        #                               Result saved) kerasTrain ->parameter reduzieren]

        # save parameters after initial training
        at.save()
        #   TODO:   [Sebastian] iv.	Save params

        # initialize noiser:
        noise_generator = Noiser(noise_type="simplistic")

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