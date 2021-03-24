from numpy import ndarray

import tensorflow as tf
from scipy.special import softmax
import numpy as np


class BayesLayer(tf.keras.layers.Layer):
    """
    This layer should be the final layer of a network
    It will apply Bayes rule to the predictions:

    Posterior = predictions / train_prior * test_prior

    Only during test phase
    """

    def __init__(self, train_prior: ndarray, test_prior: ndarray):
        """
        This layer requires the train and test priors
        :param train_prior:
        :param test_prior:
        """
        super(BayesLayer, self).__init__()
        if train_prior is None or test_prior is None:
            raise Exception('Cannot instantiate a BayesLayer without priors. Is your data loaded?')
        self.train_prior = train_prior
        self.test_prior = test_prior

    def call(self, predictions, training=None):
        """
        Apply bayes, if in test phase
        :rtype: object
        """
        if training:
            return predictions
        print('Applying bayes')
        posterior = predictions / self.train_prior * self.test_prior
        # Apply softmax to ensure that values sum to one
        posterior = tf.keras.layers.Softmax()(posterior)
        return posterior
