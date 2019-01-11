import os
from operator import concat

import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
from keras.utils import plot_model

# AS: not needed
# from ProjectFolder.Code.configuration import load_config


class LinearDecay(Callback):
    # Define a linear pattern for the decay of the learning rate.

    def __init__(self, initial_lr, epochs):
        super(LinearDecay, self).__init__()
        self.initial_lr = initial_lr
        self.decay = initial_lr / epochs

    def on_epoch_begin(self, epoch, logs={}):
        new_lr = self.initial_lr - self.decay * epoch
        print("ld: learning rate is now " + str(new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)


class HalfDecay(Callback):
    # currently not used. Was copied from keras_train.py (but not used there neither)
    # -> can probably be deleted.

    def __init__(self, initial_lr, period):
        super(HalfDecay, self).__init__()
        self.init_lr = initial_lr
        self.period = period

    def on_epoch_begin(self, epoch, logs={}):
        factor = epoch // self.period
        lr = self.init_lr / (2 ** factor)
        print("hd: learning rate is now " + str(lr))
        K.set_value(self.model.optimizer.lr, lr)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class AMTNetwork:
    def __init__(self, args):
        self.bin_multiple = args['bin_multiple']
        self.max_midi = args['max_midi']
        self.min_midi = args['min_midi']
        self.note_range = args['note_range']
        self.sr = args['sr']
        self.hop_length = args['hop_length']
        self.window_size = args['window_size']
        self.epochs = args['epochs_on_clean']

        self.feature_bins = args['feature_bins']
        self.input_shape = args['input_shape']
        self.input_shape_channels = args['input_shape_channels']

        self.bins_per_octave = 12 * self.bin_multiple  # should be a multiple of 12
        self.n_bins = args['n_bins']

        self.init_lr = args['init_lr']
        self.lr_decay = args['lr_decay']
        self.checkpoint_root = args['checkpoint_root']

        # MT: better use relu for hidden layers [http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf]
        # sigmoid for output layer

        inputs = Input(shape=self.input_shape)
        reshape = Reshape(self.input_shape_channels)(inputs)

        # normal convnet layer (have to do one initially to get 64 channels)
        conv1 = Conv2D(50, (5, 25), activation='tanh', padding='valid', data_format="channels_last")(reshape)
        do1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(1, 3))(do1)

        conv2 = Conv2D(50, (3, 5), activation='tanh', padding='valid', data_format="channels_last")(pool1)
        do2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(1, 3))(do2)

        flattened = Flatten()(pool2)
        # changed_AS
        # fc1 = Dense(1000, activation='sigmoid')(flattened)
        fc1 = Dense(500, activation='sigmoid')(flattened)
        do3 = Dropout(0.5)(fc1)

        # changed_AS
        # fc2 = Dense(200, activation='sigmoid')(do3)
        fc2 = Dense(200, activation='sigmoid')(do3)
        do4 = Dropout(0.5)(fc2)
        outputs = Dense(self.note_range, activation='sigmoid')(do4)

        self.model = Model(inputs=inputs, outputs=outputs)

        # MT: the best loss function for AMT binary_crossentropy according to 
        # [http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf]

        self.model.compile(loss='binary_crossentropy', optimizer=SGD(lr=self.init_lr, momentum=0.9), metrics=[f1,'accuracy'])
        ##MT: hier können wir auch adam nehmen statt SGD (faster) --SGD hatte , momentum=0.9
        self.model.summary()
        try:
            plot_model(self.model, to_file=os.path.join(self.checkpoint_root, 'model.png'))
        except:
            print('error: could not create png')

    def train(self, features, labels, epochs=1000, train_descr=''):
        """ Do training on the provided data set.

        """

        # filenames
        model_ckpt = os.path.join(self.checkpoint_root, train_descr)
        csv_logger = CSVLogger(os.path.join(self.checkpoint_root + train_descr + 'training.log'))

        # how does the learning rate change over time?
        if self.lr_decay == 'linear':
            decay = LinearDecay(self.init_lr, epochs)
        else:
            decay = HalfDecay(self.init_lr, 5)

        # comment SW:   checkpoint ist eine Callback Klasse, die das Model mit den Model-Parameter in eine Datei specihert.
        #               Bei der aktuellen Konfiguration wird das Modell einmal gespeichert und zwar nur das beste Validation loss.
        #               Wir müssen das Model nicht nochmal separat speichern, wenn wir diese Checkpoint-Callback implementieren.
        checkpoint_best = ModelCheckpoint(model_ckpt + '_best_weights.h5',
                                          monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        checkpoint_nth = ModelCheckpoint(model_ckpt + '_weights.{epoch:02d}-{loss:.2f}.h5', monitor='val_loss',
                                         verbose=1, mode='min', period=50)
        early_stop = EarlyStopping(patience=2, monitor='val_loss', verbose=1, mode='min')

        callbacks = [checkpoint_best, checkpoint_nth, early_stop, decay, csv_logger]

        myLoss = self.model.fit(x=features, y=labels, callbacks=callbacks, epochs=epochs, batch_size=256,
                                validation_split=0.1)

        # comment AS: Das hier ist der ursprüngliche Aufruf; die Daten werden iterativ "erzeugt" (=geladen aus den  # Files). Für uns ist das wohl nicht sinnvoll.  # history = model.fit_generator(trainGen.next(), trainGen.steps(), epochs=epochs,  #                              verbose=1, validation_data=valGen.next(), validation_steps=valGen.steps(),  #                              callbacks=callbacks)

    def transcribe(self, X):

        """ Apply learned model to data, and return the transcription.

        :param data: new data to be transcribed. Shape is (Nframes, self.window_size, self.feature_bins)
        :return: predicted transcription. Shape is (Nframes, ...)
        """

        y_pred = self.model.predict(X)
        return y_pred

    def evaluation(self, x_new, x_old, y_true):

        """ Evaluate score of predicting new noise level and compare it to score of old noise level.

                :param x_new: is x clean combined with current noise_level
                :param x_old: is x clean combined with noise level before current loop.
                :param y_true: is true labbeling of data
                :return: percentage difference of new score compared to score of noise level of anterior loop
                """

        res_new = self.model.evaluate(x_new, y_true)[1]
        res_old = self.model.evaluate(x_old, y_true)[1]
        dif = res_new - res_old
        dif_percent = dif / res_old
        #print("neues Loss", res_new)
        #print("altes Loss", res_old)
        #print("loss has increased by", dif, "absolute")
        #print("loss has increased by", dif_percent, "percent")

        return dif_percent

    def save(self, model_path):
        """
        :param model_path: String
        :type model: keras.Model
        """

        with open(model_path + ".json", "w") as json_file:
            json_file.write(self.model.to_json())
        # serialize weights to HDF5
        self.model.save_weights(model_path + ".h5")
        print("Saved trained model to disk: ", model_path)

    def load(self, model_path):
        # load json and create model
        json_file = open(model_path + '.json', 'r')
        json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(json)
        # load weights into new model
        loaded_model.load_weights(model_path + ".h5")
        print("Loaded model from disk")
        self.model = loaded_model
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(
            lr=self.init_lr))  # Sollte das laden des Modells gleich das Compilieren beinhalten? => JA.  #  Eventually compile loaded model directly in the function or to split it to the init function with IF-clause


class Noiser():
    # TODO: [Tanos] Create noise machine for several noise types to generate noise samples frame by frame.
    #               Start with Gaussian (=white), brown, pink etc.

    def __init__(self, noise_size, noise_type="simplistic"):
        self.noise_type = noise_type
        self.noise_size = noise_size
        if self.noise_type != 'simplistic':
            print(
                "WARNING: noise type " + noise_type + " not implemented. Will not generate anything!!")  # to be changed once we have other noise types...

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
