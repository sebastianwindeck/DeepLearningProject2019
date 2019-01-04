import os
import argparse
import json

'''
This script is for creating and loading a JSON structure that will hold parameters that are to be
held constant for preprocessing data for, training, and testing each model used.
'''


class DataGen:
    def __init__(self, dirpath, batch_size, args, num_files=1):
        print('initializing gen for ' + dirpath)

        self.mmdirs = os.listdir(dirpath)
        self.spe = 0  # steps per epoch
        self.dir = dirpath

        # changed_AS (added):
        self.note_range = int(args['max_midi']) - int(args['min_midi']) + 1
        self.n_bins = int(args['bin_multiple']) * note_range
        self.args = args

        for mmdir in self.mmdirs:
            print(mmdir)
            _, outputs = readmm(os.path.join(self.dir, mmdir), args)
            self.spe += len(outputs) // batch_size
            # print cnt
        self.num_files = num_files

        self.batch_size = batch_size
        self.current_file_idx = 0
        print('starting with ',
              self.mmdirs[self.current_file_idx:self.current_file_idx + self.num_files])
        for j in range(self.num_files):
            mmdir = os.path.join(self.dir, self.mmdirs[self.current_file_idx + j])
            i, o = readmm(mmdir, args)
            if j == 0:
                self.inputs, self.outputs = i, o
                print('set inputs,outputs')
            else:
                self.inputs = np.concatenate((self.inputs, i))
                self.outputs = np.concatenate((self.outputs, o))
                print('concatenated')
            self.current_file_idx = (self.current_file_idx + 1) % len(self.mmdirs)
        self.i = 0

    def steps(self):
        return self.spe

    def next(self):
        while True:
            if (self.i + 1) * self.batch_size > self.inputs.shape[0]:
                # return rest and then switch files
                x, y = self.inputs[self.i * self.batch_size:], self.outputs[self.i * self.batch_size:]
                self.i = 0
                if len(self.mmdirs) > 1:
                    # no need to open any new files if we only deal with one, like for validation
                    print('switching to ',
                          self.mmdirs[self.current_file_idx:self.current_file_idx + self.num_files])
                    for j in range(self.num_files):
                        mmdir = os.path.join(self.dir, self.mmdirs[self.current_file_idx + j])
                        # changed_AS:
                        # i,o = readmm(mmdir,args)
                        i, o = readmm(mmdir, self.args)
                        if j == 0:
                            self.inputs, self.output = i, o
                        else:
                            self.inputs = np.concatenate((self.inputs, i))
                            self.outputs = np.concatenate((self.outputs, o))
                        self.current_file_idx = (self.current_file_idx + 1) % len(self.mmdirs)

            else:
                x, y = self.inputs[self.i * self.batch_size:(self.i + 1) * self.batch_size], self.outputs[
                                                                                             self.i * self.batch_size:(
                                                                                                                                  self.i + 1) * self.batch_size]
                self.i += 1
            yield x, y


def load_config(json_fn):
    with open(json_fn, 'r') as infile:
        config = json.load(infile)
    return config

def create_config(args):
    path = os.path.join('models',args['model_name'])
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path,'config.json'), 'w') as outfile:
        json.dump(args, outfile)


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Create a config JSON')

    #possible types/values
    #model_name,spec_type,init_lr,lr_decay,bin_multiple,residual,filter_shape
    #baseline,cqt,1e-2,linear,36,False,some
    #new,logstft,1e-1,geo,96,True,full

    parser.add_argument('model_name',
                        help='model name. will create a directory for model where config,data,etc will go')
    parser.add_argument('spec_type',
                        help='Spectrogram Type, cqt or logstft')
    parser.add_argument('init_lr', type=float,
                        help='Initial Learning Rate')
    parser.add_argument('lr_decay',
                        help='How the Learning Rate Will Decay')
    parser.add_argument('bin_multiple', type=int,
                        help='Used to calculate bins_per_octave')
    parser.add_argument('residual', type=bool,
                        help='Use Residual Connections or not')
    parser.add_argument('full_window',
                        help='Whether or not the convolution window spans the full axis')

    ''' These are all constant.
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sampling Rate')
    parser.add_argument('--hl', type=int, default=512,
                        help='Hop Length')
    parser.add_argument('--ws', type=int, default=7,
                        help='Window Size')
    parser.add_argument('--bm', type=int, default=3,
                        help='Bin Multiple')
    parser.add_argument('--min', type=int, default=21, #A0
                        help='Min MIDI value')
    parser.add_argument('--max', type=int, default=108, #C8
                        help='Max MIDI value')'''

    # changed_AS
    parser.add_argument('min_midi', type=int, default=21+8+8, #A0
                        help='Min MIDI value')
    parser.add_argument('max_midi', type=int, default=108-8-8, #C8
                        help='Max MIDI value')

    args = vars(parser.parse_args())

    create_config(args)
