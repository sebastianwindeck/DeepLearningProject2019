import numpy as np
import sklearn
from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf


def opt_thresholds(y_true, y_scores):
    othresholds = np.zeros(y_scores.shape[1])
    print(othresholds.shape)
    for label, (label_scores, true_bin) in enumerate(zip(y_scores.T, y_true.T)):
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_bin, label_scores)
        max_f1 = 0
        max_f1_threshold = .5
        for r, p, t in zip(recall, precision, thresholds):
            if p + r == 0: continue
            if (2 * p * r) / (p + r) > max_f1:
                max_f1 = (2 * p * r) / (p + r)
                max_f1_threshold = t
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
        K.set_value(self.model.optimizer.lr, new_lr)


class Threshold(Callback):
    """
        decay = decay value to subtract each epoch
    """

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


class LossHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


class PredictionHistory(Callback):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.predhis = []

    def on_epoch_end(self, epoch, logs={}):
        x_train = self.train_data
        y_train = self.train_labels
        prediction = self.model.predict(x_train)
        print("Numbers of predicted notes: ", np.sum(np.round(prediction) > 0))
        print("Number of true notes: ", np.sum(np.round(y_train) > 0))
        print("Ratio of total notes played: ",
              np.round(np.sum(np.round(prediction) > 0) / (self.train_labels.shape[0] * self.train_labels.shape[1]),
                       decimals=2))  # self.predhis.append(prediction)


def f1(y_true, y_pred):
    """Calculate F1 score.
    
    :type y_true: tensorflow obeject
    :type y_pred: tensorflow obeject
    """

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        :type y_true: tensorflow obeject
        :type y_pred: tensorflow obeject
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
        :type y_true: tensorflow obeject
        :type y_pred: tensorflow obeject
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def calculating_class_weights(y_true, type='over_columns'):
    if type == 'over_all':
        # for one weight for everything
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 1])
        weights.fill(np.count_nonzero(y_true == 0) / (np.count_nonzero(y_true)))

        return weights.T

    if type == 'over_columns':
        # for one weight per pitch
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 1])  # empty array
        for i in range(number_dim):
            weights[i] = np.count_nonzero(y_true[:, i] == 0, axis=0) / (np.count_nonzero(y_true[:, i], axis=0))

        return weights.T

    else:
        print('WARNING: type is set wrong --> set to over_columns')
        calculating_class_weights(y_true=y_true, type='over_columns')


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(weighted_binary_crossentropy(y_true, y_pred, weights), axis=-1)

    return weighted_loss


def weighted_binary_crossentropy(target, output, weights, from_logits=False):
    from keras.backend.common import epsilon

    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    #  expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=weights)
