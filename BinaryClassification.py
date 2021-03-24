from datetime import datetime

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from BaseClassification import BaseClassifier
from BayesLayer import BayesLayer
from DataLoader import DataLoader
from helpers import write_log

matplotlib.use('TkAgg')


class BinaryClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.num_epochs = 50

    def get_net(self):
        """
        Get the network. Use a simple network that first expands and than shrinks the dimensionality.
        Use Leaky relu activations, the final output is given by a sigmoid
        """
        net = super().get_net()
        net.add(Dense(1, activation='sigmoid'))
        if self.apply_bayes:
            write_log('Adding Bayes layer')
            net.add(BayesLayer(self.train_prior, self.test_prior))
        return net

    def load_data(self):
        """
        Load the data using the dataloader
        """
        data_loader = DataLoader()
        data_loader.set_binary_classification()
        x, y = data_loader.load_data()
        self.split(x, y)

    def compile_net(self):
        """
        Use adam optimizer, binary cross entropy
        :return: the net
        """
        optimizer = Adam(learning_rate=.001)
        loss = tf.keras.losses.BinaryCrossentropy()
        net = self.get_net()
        net.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return net
