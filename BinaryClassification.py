import numpy as np
from sklearn.metrics import log_loss, f1_score, accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import training
import tensorflow as tf
from BaseClassification import BaseClassifier
from DataLoader import DataLoader
from helpers import write_log


class BinaryClassifier(BaseClassifier):
    """
    Base model for binary classification: htop vs background
    """

    batch_size = 64
    num_epochs = 150

    def test(self, net: training.Model, verbose=True):
        """
        Test the networks performance
        :param net the trained network
        """
        y_true = self.y_test.to_numpy()

        y_pred = net.predict(self.x_test)
        # Binary to two-class
        y_pred = np.array([[1 - y[0], y[0]] for y in y_pred])
        # Apply bayes
        if self.apply_bayes:
            y_pred = self._apply_bayes(y_pred)

        # Convert back to binary
        y_pred = np.argmax(y_pred, axis=1)

        loss = log_loss(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        if verbose:
            write_log(f'Test loss, f1, and accuracy are {loss}, {f1}, {accuracy}')
        return loss, f1, accuracy

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
        optimizer = Adam(learning_rate=.0001)
        net = self.get_net()
        net.compile(loss=self.binary_crossentropy(), optimizer=optimizer, metrics=['accuracy', self.f1])
        return net

    def binary_crossentropy(self):
        """
        loss function. Use weighted categorical crossentropy if self.class_weights is provided, else normal binary crossentropy
        @return: The loss
        """
        if self.class_weights is not None:
            return self.weighted_categorical_crossentropy(self.class_weights)
        else:
            return tf.keras.losses.BinaryCrossentropy()
