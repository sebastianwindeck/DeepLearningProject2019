
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.models import Model, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import plot_model

import numpy as np
from operator import concat
import os

import pretty_midi

import tensorflow as tf

import sklearn
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
# AS: not needed
# from ProjectFolder.Code.configuration import load_config


class LinearDecay(Callback):
    # Define a linear pattern for the decay of the learning rate.

    def __init__(self, initial_lr,epochs):
        super(LinearDecay, self).__init__()
        self.initial_lr = initial_lr
        self.decay = initial_lr/epochs

    def on_epoch_begin(self, epoch, logs={}):
        new_lr = self.initial_lr - self.decay*epoch
        print("ld: learning rate is now "+str(new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)


class HalfDecay(Callback):
    # currently not used. Was copied from keras_train.py (but not used there neither)
    # -> can probably be deleted.

    def __init__(self, initial_lr,period):
        super(HalfDecay, self).__init__()
        self.init_lr = initial_lr
        self.period = period

    def on_epoch_begin(self, epoch, logs={}):
        factor = epoch // self.period
        lr  = self.init_lr / (2**factor)
        print("hd: learning rate is now "+str(lr))
        K.set_value(self.model.optimizer.lr, lr)


class AMTNetwork:
    def __init__(self, args):
        self.bin_multiple = args['bin_multiple']
        self.max_midi = args['max_midi']
        self.min_midi = args['min_midi']
        self.note_range = args['note_range']
        self.sr = args['sr']
        self.hop_length = args['hop_length']
        self.window_size = args['window_size']

        self.feature_bins = args['feature_bins']
        self.input_shape = args['input_shape']
        self.input_shape_channels = args['input_shape_channels']

        self.bins_per_octave = 12 * self.bin_multiple  # should be a multiple of 12
        self.n_bins = args['n_bins']

        self.init_lr = args['init_lr']
        self.lr_decay = args['lr_decay']
        self.checkpoint_root = args['checkpoint_root']

    #def init_amt(self):
        # TODO: [Andreas] define network,
        #MT: better use relu for hidden layers [http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf]
        # sigmoid for output layer

        inputs = Input(shape=self.input_shape)
        reshape = Reshape(self.input_shape_channels)(inputs)

        # normal convnet layer (have to do one initially to get 64 channels)
        conv1 = Conv2D(50, (5, 25), activation='tanh')(reshape)
        do1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(1, 3))(do1)

        conv2 = Conv2D(50, (3, 5), activation='tanh')(pool1)
        do2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(1, 3))(do2)

        flattened = Flatten()(pool2)
        # changed_AS
        # fc1 = Dense(1000, activation='sigmoid')(flattened)
        fc1 = Dense(100, activation='sigmoid')(flattened)
        do3 = Dropout(0.5)(fc1)

        # changed_AS
        # fc2 = Dense(200, activation='sigmoid')(do3)
        fc2 = Dense(50, activation='sigmoid')(do3)
        do4 = Dropout(0.5)(fc2)
        outputs = Dense(self.note_range, activation='sigmoid')(do4)

        self.model = Model(inputs=inputs, outputs=outputs)

        # MT: the best loss function for AMT binary_crossentropy according to 
        # [http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf]
        self.model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.init_lr))
        ##MT: hier können wir auch adam nehmen statt SGD (faster) --SGD hatte , momentum=0.9
        self.model.summary()
        plot_model(self.model, to_file=os.path.join(self.checkpoint_root, 'model.png'))


    def train(self, features, labels, epochs, train_descr=''):
        """ Do training on the provided data set.

        """
        # TODO: [Andreas] (based on some data, "clean" or "noisy")

        # filenames
        model_ckpt = os.path.join(self.checkpoint_root + train_descr)
        csv_logger = CSVLogger(os.path.join(self.checkpoint_root + train_descr + 'training.log'))

        # how does the learning rate change over time?
        if self.lr_decay == 'linear':
            decay = LinearDecay(self.init_lr, epochs)
        else:
            decay = HalfDecay(self.init_lr, 5)

        # comment SW:   checkpoint ist eine Callback Klasse, die das Model mit den Model-Parameter in eine Datei specihert.
        #               Bei der aktuellen Konfiguration wird das Modell einmal gespeichert und zwar nur das beste Validation loss.
        #               Wir müssen das Model nicht nochmal separat speichern, wenn wir diese Checkpoint-Callback implementieren.
        checkpoint_best = ModelCheckpoint(model_ckpt + '_best_weights.{epoch:02d}-{val_loss:.2f}.h5',
                                          monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        checkpoint_nth =  ModelCheckpoint(model_ckpt + '_weights.{epoch:02d}-{val_loss:.2f}.h5',
                                          monitor='val_loss', verbose=1, mode='min', period=10)
        early_stop = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='min')

        # t = Threshold(valData)
        callbacks = [checkpoint_best, checkpoint_nth, early_stop, decay, csv_logger]

        # run a training on the data batch.
        # comment AS: to be checked!!!!
        #             does not accept callback parameters
        # comment SW: why use train_on_batch --> better use fit with callback and other params
        myLoss = self.model.train_on_batch(features, labels) #, callbacks=callbacks)

        # comment AS: Das hier ist der ursprüngliche Aufruf; die Daten werden iterativ "erzeugt" (=geladen aus den
        # Files). Für uns ist das wohl nicht sinnvoll.
        # history = model.fit_generator(trainGen.next(), trainGen.steps(), epochs=epochs,
        #                              verbose=1, validation_data=valGen.next(), validation_steps=valGen.steps(),
        #                              callbacks=callbacks)

        # comment AS: some old stuff, from keras_train. not sure whether this works with our training method, and if so
        # whether this is somehow usefull.
        '''
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('baseline/acc.png')'''

        '''
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss.png')

        # test
        testGen = DataGen(os.path.join(path, 'data', 'test'), batch_size, args)

        res = model.evaluate_generator(testGen.next(), steps=testGen.steps())
        print(model.metrics_names)
        print(res)
        '''


    def transcribe(self, data):
        """ Apply learned model to data, and return the transcription.

        :param data: new data to be transcribed. Shape is (Nframes, self.window_size, self.feature_bins)
        :return: predicted transcription. Shape is (Nframes, ...)
        """
        # TODO: [Malte] (vermutlich gibt's da eine Funktion "predict" o.ä.)
        output = []

        return output

    def evaluate(self, x_new, x_old, y_true, model):
        #x_new daten die bereits mit neuer noise verknüpft wurden, x_old daten die den letzten noise loop nicht haben
        # TODO: [Malte]
        #       allenfalls kann hier auch direkt eine Funktion in der Art evaluate(new_data, new_ground_truth)
        #       aufgerufen werden, die dann eine prediction/transcription macht und die Qualität (gem. dem
        #       festgelegten Mass) bestimmt.
        res_new = model.evaluate(x_new, y_true)
        res_old = model.evaluate(x_old, y_true)
        dif = res_new-res_old
        dif_percent = dif/res_old
        print("neues Loss",res_new)
        print("altes Loss",res_old)
        print("loss has increased by", dif, "absolute")
        print("loss has increased by", dif_percent, "percent")

        return dif_percent

    def save(self, model_path):
        """
        :param model_path: String
        :type model: keras.Model
        """

        with open(model_path+".json", "w") as json_file:
            json_file.write(self.model.to_json())
        # serialize weights to HDF5
        self.model.save_weights(model_path+".h5")
        print("Saved noise trained model", model_path, "to disk")

    def load(self, model_path):
        # load json and create model
        json_file = open(model_path+'.json', 'r')
        json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(json)
        # load weights into new model
        loaded_model.load_weights(model_path+".h5")
        print("Loaded model from disk")
        self.model = loaded_model
        # TODO: [Andreas] Sollte das laden des Modells gleich das Compilieren beinhalten?
        # Eventually compile loaded model directly in the function or to split it to the init function with IF-clause



class Noiser():
    # TODO: [Tanos] Create noise machine for several noise types and intensitiy to combine the noise frame by frame to
    #               sample

    def __init__(self, noise_size, noise_type="simplistic"):
        self.noise_type = noise_type
        self.noise_size = noise_size
        if self.noise_type != 'simplistic':
            print("WARNING: noise type " + noise_type + " not implemented. Will not generate anything!!")
            # to be changed once we have other noise types...

    def generate(self, n_noise_samples=1):
        """Generate noise samples.

        The type of the noise that will be generated, and the size of the noise array are defined by the argument given
        to the constructor.

        :param n_noise_samples: The number of noise samples to be generated.

        :return: an np.array with the specified noise
        """

        if self.noise_type == 'simplistic':
            return np.random.uniform(0, 1, size=concat([n_noise_samples], list(self.noise_size)))
        else:
            print("WARNING: noise type " + self.noise_type + " not defined. Returning 0")
            return 0
