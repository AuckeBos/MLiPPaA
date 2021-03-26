from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from BaseClassification import BaseClassifier
from DataLoader import DataLoader


class BinaryClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.batch_size = 256
        self.num_epochs = 50

    def get_net(self):
        """
        Get the network. Use a simple network that first expands and than shrinks the dimensionality.
        Use Leaky relu activations, the final output is given by a sigmoid
        """
        net = super().get_net()
        net.add(Dense(1, activation='sigmoid'))
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
        net.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', self.f1])
        return net
