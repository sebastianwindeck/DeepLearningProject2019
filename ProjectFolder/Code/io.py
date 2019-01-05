import librosa
import numpy as np
import os

# TODO: [Tanos] Copy and configurate the inputs and outputs
#               1. input for .mid
#               2. input for .wav
#               3. creation of .wav files
#                   Please add missing methods concerning the inputs or outputs

# Andreas: Habe hier die Funktionen zum Einlesen der Daten und zur Extraktion der cqt-Features eingefügt (das meiste ist
#          kopiert / adaptiert aus advTrain

'''
Dieses File enthält diverse Funktionen zur Extraktion von Audio-Features im weiteren Sinne.
'''


def extract_features(audio_filename, args):
    """ Extract audio feature

    This function was previously called wav2inputnp in advTrain.py.

    :param audio_filename: name of an audio file (.wav format) for which the features are to be calculated.
    :param args: a dictionary containing the parameters. This function needs the parameters 'featType'
    :return: an np.ndarray with the requested features. shape[0] is the time dimension
    """
    print("extract_features")
    spec_type = args['spec_type']

    if spec_type == 'cqt':
        bin_multiple = args['bin_multiple']
        max_midi = args['max_midi']
        min_midi = args['min_midi']
        note_range = max_midi - min_midi
        sr = args['sr']
        hop_length = args['hop_length']
        window_size = args['window_size']

        bins_per_octave = 12 * bin_multiple  # should be a multiple of 12
        n_bins = note_range * bin_multiple

        # down-sample,mono-channel
        y, _ = librosa.load(audio_filename, sr)
        # y: an np.ndarray[ shape=(n,) ] giving the audio time series. librosa.load automatically downsamples to the
        # required sample rate sr
        # doku zu librosa.cqt:
        # https://librosa.github.io/librosa/generated/librosa.core.cqt.html?highlight=cqt#librosa.core.cqts
        S = librosa.cqt(y, fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins)
        S = S.T

        # AS: ich bin nicht sicher, was hier sinnvoller ist: abs oder amplitude_to_db. Das war so im preprocess.py von
        # wav2mid-master. db ist eine logarithmische Skala
        # S = librosa.amplitude_to_db(S)
        S = np.abs(S)
        min_db = np.min(S)
        print(np.min(S), np.max(S), np.mean(S))
        S = np.pad(S, ((window_size // 2, window_size // 2), (0, 0)), 'constant', constant_values=min_db)

        windows = []

        # IMPORTANT NOTE:
        # Since we pad the the spectrogram frame,
        # the onset frames are actually `offset` frames.
        # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
        # starting at frame 0 of the padded spectrogram
        for i in range(S.shape[0] - window_size + 1):
            w = S[i:i + window_size, :]
            windows.append(w)

        # print inputs
        x = np.array(windows)
        return x

    else:
        print("WARNING: feature type " + spec_type + " not implemented.")
        return 0


def mid2outputnp(pm_mid, times):
    """ TODO: Add comment
    """
    piano_roll = pm_mid.get_piano_roll(fs=sr, times=times)[min_midi:max_midi + 1].T
    piano_roll[piano_roll > 0] = 1
    return piano_roll


def join_create_path(base_path, sub_dir):
    """ Combines the base_path with sub_dir and creates the new directory if it does not already exist.

    :param base_path: The basis directory.
    :param sub_dir: The subdirectory to be added to base_path
    :return: The full new directory
    """
    new_path = os.path.join(base_path, sub_dir)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def prepareData(args):
    """Prepare the data for the training / testing / etc.
    Starting from the root directory of the maps dataset (given as args['wav_dir']), all audio files (.wav) with a
    corresponding .mid file are processed and concatenated. Both the feature arrays and the correct transcriptions are
    both stored to file and returned.

    :param args: parameter dictionary, containing all the parameters needed for feature extraction (and others).

    :return: inputs, outputs, path.
        inputs are the extracted features of the audio files. shape[0] is the dimension along with all features are
            concatenated.
        output are the transcriptions (with the same sample rate as the inputs). Also here, shape[0] is the dimension
            along with all transcriptions (the ground truth for the amt) are concatenated.
        path: The path where the two data files are stored. The corresponding files are path/input.dat and
            path/output.dat
    """
    print("Starting preprocessing")

    # params
    max_midi = args['max_midi']
    min_midi = args['min_midi']
    note_range = max_midi - min_midi
    window_size = args['window_size']
    sr = args['sr']
    hop_length = args['hop_length']
    wav_dir = args['wav_dir']

    path = os.path.join('models', args['model_name'])
    # config = load_config(os.path.join(path,'config.json'))

    bin_multiple = int(args['bin_multiple'])

    framecnt = 0
    maxFramesPerFile = args['maxFramesPerFile']
    maxFrames = args['maxFrames']

    fileappend = str(maxFramesPerFile) + 'pf_max' + str(maxFrames) + '.dat'

    datapath = join_create_path(path, 'data')
    filenameIN = os.path.join(datapath, 'input_' + fileappend)
    filenameOUT = os.path.join(datapath, 'output_' + fileappend)

    if os.path.isfile(filenameIN) and os.path.isfile(filenameOUT):
        n_bins = note_range * bin_multiple
        print('loading precomputed data from ' + filenameIN)
        mmi = np.memmap(filenameIN, mode='r')
        inputs = np.reshape(mmi, (-1, window_size, n_bins))

        mmo = np.memmap(filenameOUT, mode='r')
        outputs = np.reshape(mmo, (-1, note_range))

        return inputs, outputs, path

    # hack to deal with high PPQ from MAPS
    # https://github.com/craffel/pretty-midi/issues/112
    pretty_midi.pretty_midi.MAX_TICK = 1e10

    inputs, outputs = [], []
    addCnt, errCnt = 0, 0

    for s in os.listdir(wav_dir):
        subdir = os.path.join(wav_dir, s)
        if not os.path.isdir(subdir):
            continue
        # recursively search in subdir
        print(subdir)
        for dp, dn, filenames in os.walk(subdir):
            # in each level of the directory, look at filenames ending with .mid
            for f in filenames:
                # if there exists a .wav file and .midi file with the same name

                if f.endswith('.wav'):
                    audio_filename = f
                    fprefix = audio_filename.split('.wav')[0]
                    mid_fn = fprefix + '.mid'
                    txt_fn = fprefix + '.txt'
                    print("Handling files {}".format(fprefix))
                    if mid_fn in filenames:
                        # extract_features
                        audio_filename = os.path.join(dp, audio_filename)
                        inputnp = extract_features(audio_filename,
                                              spec_type=args['spec_type'],
                                              bin_multiple=int(args['bin_multiple']))
                        times = librosa.frames_to_time(np.arange(inputnp.shape[0]),
                                                       sr=sr, hop_length=hop_length)
                        # mid2outputnp
                        mid_fn = os.path.join(dp, mid_fn)
                        pm_mid = pretty_midi.PrettyMIDI(mid_fn)
                        outputnp = mid2outputnp(pm_mid, times)

                        # check that num onsets is equal
                        if inputnp.shape[0] == outputnp.shape[0]:
                            print("adding to dataset fprefix {}".format(fprefix))
                            addCnt += 1
                            if maxFramesPerFile > 0 and inputnp.shape[0] > maxFramesPerFile:
                                inputnp = inputnp[:maxFramesPerFile]
                                outputnp = outputnp[:maxFramesPerFile]
                            framecnt += inputnp.shape[0]
                            print("framecnt is {}".format(framecnt))
                            inputs.append(inputnp)
                            outputs.append(outputnp)
                        else:
                            print("error for fprefix {}".format(fprefix))
                            errCnt += 1
                            print(inputnp.shape)
                            print(outputnp.shape)

                if maxFrames > 0 and framecnt > maxFrames:
                    print("have enought frames, leaving {}".format(subdir))
                    break
            if maxFrames > 0 and framecnt > maxFrames:
                print("have enought frames, leaving {}".format(wav_dir))
                break

        if maxFrames > 0 and framecnt > maxFrames:
            print("have enought frames, leaving {}".format(wav_dir))
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
        # fn += '.h5'
        # save inputs,outputs to hdf5 file
        # fnpath = join_create_path(datapath,fn)

        mmi = np.memmap(filename=filenameIN, mode='w+', shape=inputs.shape)
        mmi[:] = inputs[:]
        mmo = np.memmap(filename=filenameOUT, mode='w+', shape=outputs.shape)
        mmo[:] = outputs[:]
        del mmi
        del mmo

    return inputs, outputs, path

# End audio stuff