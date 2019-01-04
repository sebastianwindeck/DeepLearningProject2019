from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model
import os
import matplotlib.pyplot as plt
from config import load_config




class AMTNetwork():

    def __init__(self):
        # TODO: [Andreas] define network,
        pass

    def baseline_model(self):
        # TODO: [Andreas] build network architecture
        inputs = Input(shape=input_shape)
        reshape = Reshape(input_shape_channels)(inputs)

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
        outputs = Dense(note_range, activation='sigmoid')(do4)

        model = Model(inputs=inputs, outputs=outputs)
        return model
        # 	Festlegen: Parameter Anzahl
        #   	Architektur: Tiefe
        # 	    Länge Samples
        # 	    Anzahl Samples

        

    def train(self, data, type):
        # TODO: [Andreas] (based on some data, "clean" or "noisy")
        path = os.path.join('models', args['model_name'])
        config = load_config(os.path.join(path, 'config.json'))

        global feature_bins
        global input_shape
        global input_shape_channels

        bin_multiple = int(args['bin_multiple'])
        # changed_AS:
        # note_range = int(args['max_midi']) - int(args['min_midi']) + 1
        print('bin multiple', str(np.log2(bin_multiple)))
        feature_bins = note_range * bin_multiple
        input_shape = (window_size, feature_bins)
        input_shape_channels = (window_size, feature_bins, 1)

        # filenames
        model_ckpt = os.path.join(path, 'ckpt.h5')

        # train params
        batch_size = 256
        # epochs = 1000
        # changed_AS:
        epochs = 5

        trainGen = DataGen(os.path.join(path, 'data', 'train'), batch_size, args)
        valGen = DataGen(os.path.join(path, 'data', 'val'), batch_size, args)
        # valData = load_data(os.path.join(path,'data','val'))
        
        ###Hier Model aussuchen

        if os.path.isfile(model_ckpt):
            print('loading model')
            model = load_model(model_ckpt)
        else:
            model = baseline_model()

        init_lr = float(args['init_lr'])

        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=init_lr, momentum=0.9))
        model.summary()
        plot_model(model, to_file=os.path.join(path, 'model.png'))

        checkpoint = ModelCheckpoint(model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='min')
        # tensorboard = TensorBoard(log_dir='./logs/baseline/', histogram_freq=250, batch_size=batch_size)
        if args['lr_decay'] == 'linear':
            decay = linear_decay(init_lr, epochs)
        else:
            decay = half_decay(init_lr, 5)
        csv_logger = CSVLogger(os.path.join(path, 'training.log'))
        # t = Threshold(valData)
        callbacks = [checkpoint, early_stop, decay, csv_logger]

        history = model.fit_generator(trainGen.next(), trainGen.steps(), epochs=epochs,
                                      verbose=1, validation_data=valGen.next(), validation_steps=valGen.steps(),
                                      callbacks=callbacks)

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        '''plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('baseline/acc.png')'''

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss.png')

        pass

    def transcribe(self, data):
        # TODO: [Andreas] (apply learned network to new data and generate output)
        output = []

        return output

    def evaluate(self, y_pred, y_true):
        # TODO: [Andreas]   (compare transcription with “ground truth”)
        # test
        testGen = DataGen(os.path.join(path, 'data', 'test'), batch_size, args)

        res = model.evaluate_generator(testGen.next(), steps=testGen.steps())
        print(model.metrics_names)
        print(res)

        return

    def save(self):
        # TODO: [Sebastian] Save model to output file after learning
        pass


class Noiser():
    # TODO: [Tanos] Create noise machine for several noise types and intensitiy to combine the noise frame by frame to sample

    def __init__(self):
        pass

    def calculator(self):
        pass


def featureextraction():
    # TODO: [Andreas] (basierend auf CQT aus librosa)
    output = []

    return output
