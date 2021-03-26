import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam

from BaseClassification import BaseClassifier
from BayesLayer import BayesLayer
from DataLoader import DataLoader
from helpers import write_log
from tensorflow.python.keras.engine import training
from sklearn.metrics import accuracy_score
import numpy as np


class MultiClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.num_epochs = 150

    def get_net(self):
        """
         Get the network. Use a simple network that first expands and than shrinks the dimensionality.
         Use Leaky relu activations, the final output is given by a softmax
         """
        net = super().get_net()
        net.add(Dense(5))
        net.add(Softmax())
        return net

    def load_data(self):
        """
        Load the data using the dataloader
        """
        data_loader = DataLoader()
        x, y = data_loader.load_data()
        self.split(x, y)

    def compile_net(self):
        """
        Use adam optimizer, categorical cross entropy
        :return: the net
        """

        optimizer = Adam(learning_rate=.0001)
        loss = tf.keras.losses.CategoricalCrossentropy()
        net = self.get_net()
        net.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', self.f1])
        return net

    def test_binary(self, net: training.Model):
        """
        Test the network, print loss and accuracy
        :param net the trained network
        """
        y_pred = net.predict(self.x_test)
        prior = np.array(self.test_prior)
        y = self.y_test
        y_classes = y['4top']
        y_binary = [int(y == 1) for y in y_classes]
        y_pred_binary = []
        for y in y_pred:
            y = np.array(y)
            prior_subtracted = y - prior
            percentage_above_priors = prior_subtracted / prior
            class_of_highest_prob = np.argmax(percentage_above_priors, axis=0)
            prediction = int(class_of_highest_prob == 0)
            y_pred_binary.append(prediction)
        # pred_true = sum(y_pred_binary)
        # pred_false = len(y_pred_binary) - pred_true
        # actual_true = sum(y_binary)
        # actual_false = len(y_binary) - actual_true

        accuracy = accuracy_score(y_binary, y_pred_binary)
        write_log(f'The test accuracy of binary classification is {accuracy}')
