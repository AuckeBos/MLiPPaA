import abc
from datetime import datetime

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import training

import tensorflow as tf
from EvaluationCallback import EvaluationCallback


class BaseClassifier(abc.ABC):
    """"
    Base class for classifiers
    """
    # Log location for tensorboard
    logdir: str
    # Batch size for training
    batch_size: int
    # #Epochs for training
    num_epochs: int

    # Training callbacks
    callbacks: []

    # Data
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_val: pd.DataFrame
    y_val: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame

    # Data iterators, used during training and evaluatrion
    train_iter: Dataset
    val_iter: Dataset
    test_iter: Dataset

    # Priors
    train_prior: ndarray = None
    test_prior: ndarray = None

    # If true, add a bayes layer during test time
    apply_bayes = False

    def __init__(self):
        """
        Add tensorflow callbacks
        - Tensorboard provides visualization during training
        - EarlyStopping ensures we don't train for too long, preventing overfitting
        """
        self.logdir = f'./tensorboard/{datetime.now().strftime("%m-%d %H:%M")}'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
        self.callbacks = [tensorboard, es]

    def compute_priors(self, train, test):
        """
        Compute the priors on train and test data
        :param train:  The train data
        :param test:  The test data
        """
        train = train.value_counts().array
        test = test.value_counts().array
        train_sum = np.sum(train)
        test_sum = np.sum(test)
        self.train_prior = np.array([x / train_sum for x in train]) * len(train)
        self.test_prior = np.array([x / test_sum for x in test]) * len(test)

    def split(self, x, y):
        """
        - Split data into train, val, test (70%, 20%, 10%)
        - Compute and save priors
        - Add the EvaluationCallback to this.callbacks.
            This callback evaluates the test set after each epoch, and adds the current accuracy and loss to tensorboard.
            The results are ofcourse not used for training, but rather to show progress on the test set during training
        """
        # Split data
        x_train_val, self.x_test, y_train_val, self.y_test = train_test_split(x, y, test_size=0.1)
        # Compute priors
        self.compute_priors(y_train_val, self.y_test)
        # Add Evaluation callback
        self.callbacks.append(EvaluationCallback(self.x_test, self.y_test, self.logdir))
        # Split train_val into train and val
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train_val, y_train_val, test_size=.2 / 0.9)

        # Create iterators
        self.train_iter = tf.data.Dataset.from_tensor_slices((self.x_train.values, self.y_train.values)).batch(self.batch_size).shuffle(self.x_train.shape[0])
        self.val_iter = tf.data.Dataset.from_tensor_slices((self.x_val.values, self.y_val.values)).batch(self.batch_size)
        self.test_iter = tf.data.Dataset.from_tensor_slices((self.x_test.values, self.y_test.values)).batch(self.batch_size)

    @abc.abstractmethod
    def get_net(self) -> training.Model:
        """
        Returns a base network, must be extended by specific implementations
        Final layer returns RELU activation after Dense(8)
        :rtype: training.Model
        """
        weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1)
        net = Sequential()
        net.add(Dense(256, input_dim=164, kernel_initializer=weight_initializer))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(512))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(1024))
        net.add(Dropout(0.2))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(512))
        net.add(Dropout(0.2))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(256))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(128))
        net.add(Dropout(0.4))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(64))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(32))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(16))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(8))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        return net

    @abc.abstractmethod
    def load_data(self):
        """
        Load the data using the dataloader
        """
        pass

    @abc.abstractmethod
    def compile_net(self):
        """
        Compile the net
        """
        pass

    def train(self):
        """
        Train the network
        - Visualize result using tensorboard
        - Use verbose output to show accuracy and loss during training
        :return the trained network
        """
        self.load_data()
        net = self.compile_net()

        history = net.fit(
            self.train_iter,
            validation_data=self.val_iter,
            epochs=self.num_epochs,
            verbose=1,
            callbacks=self.callbacks,
        )
        return net

    def test(self, net: training.Model):
        """
        Test the network, print loss and accuracy
        :param net the trained network
        """
        results = net.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)
