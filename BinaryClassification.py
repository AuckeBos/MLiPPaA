from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from BaseClassification import BaseClassifier
from DataLoader import DataLoader


class BinaryClassifier(BaseClassifier):
    """
    Base model for binary classfication: htop vs background
    """

    batch_size = 256
    num_epochs = 50

    def get_net(self):
        """
        Get the network. Use bas network and append Dense(1) with sigmoid
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
