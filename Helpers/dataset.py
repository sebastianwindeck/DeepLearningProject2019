import tensorflow as tf
import numpy as np
import datetime

import pickle
from six.moves import cPickle

"""
ideas from 
tensorflow.org tensorflow/contrib/learn/python/learn/datasets/mnist.py
and
https://github.com/qiuqiangkong/music_transcription_MAPS
"""

class DataSet(object):
    def __init__(self, samples):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._samples = samples 
        self._n_samples = samples.shape[0]

    def scale_on_x_list(x_list, scaler):
        """Scale list of ndarray. 
        """
        return [scaler.transform(e) for e in x_list]

    def get_next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` samples from this data set."""
        """originally from tensorflow.org tensorflow/contrib/learn/python/learn/datasets/mnist.py"""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = np.arange(self._n_samples)
          np.random.shuffle(perm0)
          waves = self._samples[perm0]
        # Go to the next epoch
        if start + batch_size > self._n_samples:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest samples in this epoch
          rest_n_samples = self._n_samples - start
          y_rest_part = waves[start:self._n_samples]
          # Shuffle the data
          if shuffle:
            perm = np.arange(self._n_samples)
            np.random.shuffle(perm)
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_num_samples
          end = self.index_in_epoch
          y_new_part = waves[start:end]
          return np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._samples[start:end]

