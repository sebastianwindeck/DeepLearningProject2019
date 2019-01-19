import os
from operator import concat

import numpy as np
from acoustics.generator import white, pink, blue, brown, violet
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from keras.utils import plot_model
from model_functions import LinearDecay, HalfDecay, PredictionHistory
from model_functions import calculating_class_weights, get_weighted_loss, f1
from visualize import visualize_weights


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
        fc1 = Dense(1000, activation='sigmoid')(flattened)
        do3 = Dropout(0.5)(fc1)

        fc2 = Dense(200, activation='sigmoid')(do3)
        do4 = Dropout(0.5)(fc2)
        outputs = Dense(self.note_range, activation='sigmoid')(do4)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compilation(self, y_true, save_path):

        # plot balancing weight and save to file
        visualize_weights(y_true, save_path)

        # compile model and visualize
        self.model.compile(loss=get_weighted_loss(calculating_class_weights(y_true, type='over_columns')),
                           optimizer=SGD(lr=self.init_lr, momentum=0.9), metrics=[f1])
        self.model.summary()
        try:
            plot_model(self.model, to_file=os.path.join(self.checkpoint_root, 'model.png'))

        except:
            print('Error: Could not create png')

    def train(self, features, labels, epochs=1000, train_descr=''):
        """
        Do training on the provided data set.

        """

        batch_size = 256

        # filenames
        model_ckpt = os.path.join(self.checkpoint_root, train_descr)
        csv_logger = CSVLogger(os.path.join(self.checkpoint_root, train_descr + 'training.log'))
        pred_hist = PredictionHistory(features, labels)

        if self.lr_decay == 'linear':
            decay = LinearDecay(self.init_lr, epochs)
        else:
            decay = HalfDecay(self.init_lr, 5)

        checkpoint_best = ModelCheckpoint(model_ckpt + '_best_weights.h5', monitor='val_loss', verbose=0,
                                          save_best_only=True, mode='min')
        early_stop = EarlyStopping(patience=30, monitor='val_loss', verbose=0, mode='min')

        callbacks = [checkpoint_best,  # checkpoint_nth,
                     early_stop, decay, csv_logger, pred_hist]
        self.model.fit(x=features, y=labels, callbacks=callbacks, epochs=epochs, batch_size=batch_size,
                       validation_split=0.2, verbose=2)

    def transcribe(self, x):

        """ Apply learned model to data, and return the transcription.
        :param x: new data to be transcribed. Shape is (Nframes, self.window_size, self.feature_bins)
        :return: predicted transcription. Shape is (Nframes, ...)
        """

        y_pred = self.model.predict(x)
        return y_pred

    def getscores(self, x, y):
        score = self.model.evaluate(x, y)
        return score

    def evaluate_old(self, x_old, y):
        score = self.model.evaluate(x_old, y)[0]
        return score

    def evaluation(self, x_new, res_old, y_true):

        """ Evaluate score of predicting new noise level and compare it to score of old noise level.

                :param x_new: is x clean combined with current noise_level
                :param res_old: is old evaluation from base level set
                :param y_true: is true labbeling of data
                :return: percentage difference of new score compared to score of noise level of anterior loop
                """

        res_new = self.model.evaluate(x_new, y_true)[0]
        print("new score", res_new)
        # print('Number of true notes: ', np.count_nonzero(y_true))
        # print('Number of predicted clean notes: ', np.count_nonzero(self.model.predict(x_old)))
        # print('Number of predicted noisy notes: ', np.count_nonzero(self.model.predict(x_new)))

        print("old score", res_old)
        dif = res_new - res_old
        dif_percent = dif / res_old
        # print("new  Loss", res_new)
        # print("old Loss", res_old)
        # print("loss has increased by", dif, "absolute")
        # print("loss has increased by", dif_percent, "percent")

        return dif_percent

    def save(self, model_path):
        """
        :param model_path: String
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


# Generator for sample batches
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


# Noise generator
class Noiser:

    def __init__(self, noise_size, noise_type="simplistic"):
        self.noise_type = noise_type
        self.noise_size = noise_size
        if self.noise_type.lower() not in {'simplistic', 'gaussian', 'white', 'normal', 'pink', 'blue', 'brown',
                                           'violet'}:
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
            return np.reshape(np.zeros(n), s)
