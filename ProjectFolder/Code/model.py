import os
from operator import concat

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from keras.utils import plot_model

import sklearn
from sklearn.utils.class_weight import compute_class_weight

from acoustics.generator import white, pink,blue, brown, violet

import matplotlib.pyplot as plt


def hn_multilabel_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def opt_thresholds(y_true, y_scores):
    othresholds = np.zeros(y_scores.shape[1])
    print(othresholds.shape)
    for label, (label_scores, true_bin) in enumerate(zip(y_scores.T, y_true.T)):
        # print label
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_bin, label_scores)
        max_f1 = 0
        max_f1_threshold = .5
        for r, p, t in zip(recall, precision, thresholds):
            if p + r == 0: continue
            if (2 * p * r) / (p + r) > max_f1:
                max_f1 = (2 * p * r) / (p + r)
                max_f1_threshold = t
        # print label, ": ", max_f1_threshold, "=>", max_f1
        othresholds[label] = max_f1_threshold
        print(othresholds)
    return othresholds


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


class Threshold(Callback):
    '''
        decay = decay value to subtract each epoch
    '''

    def __init__(self, val_data):
        super(Threshold, self).__init__()
        self.val_data = val_data
        _, y = val_data
        print(y)
        self.othresholds = np.full(y.shape[1], 0.5)

    def on_epoch_end(self, epoch, logs={}):
        # find optimal thresholds on validation data
        x, y_true = self.val_data
        y_scores = self.model.predict(x)
        self.othresholds = opt_thresholds(y_true, y_scores)
        y_pred = y_scores > self.othresholds
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
        print("validation p,r,f,s:")
        print(p, r, f, s)


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

def calculating_class_weights(y_true):
    print(np.unique(y_true))
    number_dim = np.shape(y_true)[1]  # columns
    weights = np.empty([number_dim, 2])  # empty array
    for i in range(number_dim):
        print(i)
        #print(np.unique(y_true[:, i]))
        try:
            weights[i] = compute_class_weight('balanced', classes=[0, 1], y=y_true[:, i])
        except:
            weights[i] = np.array([1, 1])
        print(i, ": ", weights[i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis=-1)

    return weighted_loss



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
        self.balance_classes = args['balance_classes']

        # MT: better use relu for hidden layers [http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf]
        # sigmoid for output layer

        inputs = Input(shape=self.input_shape)
        reshape = Reshape(self.input_shape_channels)(inputs)

        # normal convnet layer (have to do one initially to get 64 channels)
        conv1 = Conv2D(50, (5, 25), activation='relu', padding='same', data_format="channels_last")(reshape)
        do1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(1, 3))(do1)

        conv2 = Conv2D(50, (3, 5), activation='relu', padding='same', data_format="channels_last")(pool1)
        do2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(1, 3))(do2)

        flattened = Flatten()(pool2)
        # changed_AS
        # fc1 = Dense(1000, activation='sigmoid')(flattened)

        fc1 = Dense(400, activation='sigmoid')(flattened)
        do3 = Dropout(0.5)(fc1)

        # changed_AS
        # fc2 = Dense(200, activation='sigmoid')(do3)
        fc2 = Dense(100, activation='sigmoid')(do3)
        do4 = Dropout(0.5)(fc2)
        outputs = Dense(self.note_range, activation='sigmoid')(do4)

        self.model = Model(inputs=inputs, outputs=outputs)
        # MT: the best loss function for AMT binary_crossentropy according to
        # [http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf]


    def compilation(self, y_true):
        weights = calculating_class_weights(y_true=y_true)
        self.model.compile(loss=get_weighted_loss(weights), optimizer=SGD(lr=self.init_lr, momentum=0.9), metrics=[f1])
        ##MT: hier können wir auch adam nehmen statt SGD (faster) --SGD hatte , momentum=0.9
        self.model.summary()
        try:
            plot_model(self.model, to_file=os.path.join(self.checkpoint_root, 'model.png'))
        except:
            print('error: could not create png')


    def train(self, features, labels, args, epochs=1000, train_descr=''):
        """
        Do training on the provided data set.

        """

        batch_size = 256

        # trainGen = Generator(features,labels, batch_size, args)
        # valGen = Generator(features, labels, batch_size, args)

        # filenames
        model_ckpt = os.path.join(self.checkpoint_root, train_descr)
        csv_logger = CSVLogger(os.path.join(self.checkpoint_root, train_descr + 'training.log'))

        if self.lr_decay == 'linear':
            decay = LinearDecay(self.init_lr, epochs)
        else:
            decay = HalfDecay(self.init_lr, 5)

        # comment SW:   checkpoint ist eine Callback Klasse, die das Model mit den Model-Parameter in eine Datei specihert.
        #               Bei der aktuellen Konfiguration wird das Modell einmal gespeichert und zwar nur das beste Validation loss.
        #               Wir müssen das Model nicht nochmal separat speichern, wenn wir diese Checkpoint-Callback implementieren.
        checkpoint_best = ModelCheckpoint(model_ckpt + '_best_weights.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')
        # checkpoint_nth = ModelCheckpoint(model_ckpt + '_weights.{epoch:02d}-{loss:.2f}.h5', monitor='val_loss',
        # verbose=1, mode='min', period=50)
        early_stop = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='min')

        callbacks = [checkpoint_best,  # checkpoint_nth,
                     early_stop, decay, csv_logger]

        # class_weights = NULL
        # self.model.fit(x=features, y=labels, callbacks=callbacks, epochs=epochs, batch_size=batch_size,
        # validation_split=0.1, class_weight=class_weights)
        self.model.fit(x=features, y=labels, callbacks=callbacks, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1)

        # self.model.fit_generator(generator=next(trainGen),
        #                         steps_per_epoch=trainGen.steps, epochs=epochs,
        #                        verbose=1,validation_data=next(valGen), validation_steps=valGen.steps,callbacks=callbacks)

        # comment AS: Das hier ist der ursprüngliche Aufruf; die Daten werden iterativ "erzeugt" (=geladen aus den
        # Files). Für uns ist das wohl nicht sinnvoll.
        # history = model.fit_generator(trainGen.next(), trainGen.steps(), epochs=epochs,
        #                              verbose=1, validation_data=valGen.next(), validation_steps=valGen.steps(),
        #                              callbacks=callbacks)

    def transcribe(self, x):

        """ Apply learned model to data, and return the transcription.

        :param x: new data to be transcribed. Shape is (Nframes, self.window_size, self.feature_bins)
        :return: predicted transcription. Shape is (Nframes, ...)
        """

        y_pred = self.model.predict(x)
        return y_pred

    def evaluation(self, x_new, x_old, y_true):

        """ Evaluate score of predicting new noise level and compare it to score of old noise level.

                :param x_new: is x clean combined with current noise_level
                :param x_old: is x clean combined with noise level before current loop.
                :param y_true: is true labbeling of data
                :return: percentage difference of new score compared to score of noise level of anterior loop
                """

        res_new = self.model.evaluate(x_new, y_true)[0]
        print('Number of true notes: ', np.count_nonzero(y_true))
        print('Number of predicted clean notes: ', np.count_nonzero(self.model.predict(x_old)))
        print('Number of predicted noisy notes: ', np.count_nonzero(self.model.predict(x_new)))
        res_old = self.model.evaluate(x_old, y_true)[0]
        dif = res_new - res_old
        dif_percent = dif / res_old
        # print("neues Loss", res_new)
        # print("altes Loss", res_old)
        # print("loss has increased by", dif, "absolute")
        # print("loss has increased by", dif_percent, "percent")

        return dif

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
        with tf.device('/cpu:0'):
            self.model = loaded_model


class Generator:

    def __init__(self, features, labels, batch_size, args):
        print('Initialize the Generator')

        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.window_size = args['window_size']

        self.steps = features.shape[0] // batch_size
        self.i = 0

    def __next__(self):
        # Create empty arrays to contain batch of features and labels#
        batch_features = np.zeros((self.batch_size, self.steps, self.window_size, 168))
        batch_labels = np.zeros((self.batch_size, self.steps, 56))
        while True:
            for i in range(self.batch_size):
                index = np.random.choice(self.features.shape[0], self.steps, replace=False)
                batch_features[i] = self.features[index]
                batch_labels[i] = self.labels[index]
            yield batch_features[self.i], batch_labels[self.i]

            self.i += 1


class Noiser():

    def __init__(self, noise_size, noise_type="simplistic"):
        self.noise_type = noise_type
        self.noise_size = noise_size
        if self.noise_type.lower()  not in {'simplistic', 'gaussian', 'white', 'normal', 'pink', 'blue', 'brown', 'violet'} :
            print("WARNING: noise type " + noise_type + " not implemented. Will not generate anything!!")

    def generate(self, n_noise_samples=1):
        """Generate noise samples.

        The type of the noise that will be generated, and the size of the noise array are defined by the argument given
        to the constructor.

        :param n_noise_samples: The number of noise samples to be generated.

        :return: an np.array with the specified noise
        """

        n = n_noise_samples * self.noise_size[0] * self.noise_size[1]
        s = concat([n_noise_samples], list(self.noise_size))
        if self.noise_type == 'simplistic':
            return np.random.uniform(0, 1, size=concat([n_noise_samples], list(self.noise_size)))
        elif self.noise_type.lower() in {'gaussian', 'white', 'normal'}:
            return np.reshape(white(n), s)
        elif self.noise_type.lower() == 'pink':
            return np.reshape(pink(n), s)
        elif self.noise_type.lower() == 'blue':
            return np.reshape(blue(n), s)
        elif self.noise_type.lower() == 'brown':
            return np.reshape(brown(n), s)
        elif self.noise_type.lower() == 'violet':
            return np.reshape(violet(n), s)
        else:
            print("WARNING: noise type " + self.noise_type + " not defined. Returning 0")
            return np.reshape(np.zeros((n)), s)

