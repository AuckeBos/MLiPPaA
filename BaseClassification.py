import abc
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from numpy import ndarray
from sklearn.metrics import log_loss, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.engine import training

import tensorflow as tf
from EvaluationCallback import EvaluationCallback
from helpers import write_log
from matplotlib import pyplot as plt


class BaseClassifier(abc.ABC):
    """"
    Base class for classifiers. Provides functionality that is used in all or some specific implementations
    - Base network
    - Loss and accuracy functions
    - Data processing

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
    x_train: (pd.DataFrame, ndarray)
    y_train: pd.DataFrame
    x_val: (pd.DataFrame, ndarray)
    y_val: pd.DataFrame
    x_test: (pd.DataFrame, ndarray)
    y_test: pd.DataFrame

    # If is not none, use these weights for weighted cross entropy loss
    class_weights = None

    # Priors
    train_prior: ndarray = None
    test_prior: ndarray = None

    # If true apply bayes during test phase
    apply_bayes = False

    # If this value is not None, rebalance the test set to this distribution, by down-sampling the majority classes
    # Format should be [percent_of_class1, percent_of_class2, ...], percentages in range (0,1)
    rebalance_test: ndarray = None

    # If this value is not None, rebalance the train and validation set to this distribution, by down-sampling the majority classes
    # Format should be [percent_of_class1, percent_of_class2, ...], percentages in range (0,1)
    rebalance_train_val: ndarray = None

    def __init__(self):
        """
        Add tensorflow callbacks
        - Tensorboard provides visualization during training
        - EarlyStopping ensures we don't train for too long, preventing overfitting
        - Reduce lr on plateau reduces lr when loss is not decreasing anymore
        """
        self.logdir = f'./tensorboard/{datetime.now().strftime("%m-%d %H:%M")}'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        es = EarlyStopping(patience=8, verbose=1)
        reduce_lr_on_plateau = ReduceLROnPlateau(factor=.4, patience=3, verbose=1)
        self.callbacks = [tensorboard, es, reduce_lr_on_plateau]

    def compute_priors(self, train, test):
        """
        Compute the priors on train and test data
        :param train:  The train data
        :param test:  The test data
        """
        # If class count is 2, we sort ascending: priors are in form [class=0, class=1]
        # Else sort descending, such that we have [class=[1,0,0,0,..], class=[0,1,0,0,0...]]
        ascending = len(test.value_counts()) == 2

        train = train.value_counts().sort_index(ascending=ascending).array
        test = test.value_counts().sort_index(ascending=ascending).array
        train_sum = np.sum(train)
        test_sum = np.sum(test)
        self.train_prior = np.array([x / train_sum for x in train])
        self.test_prior = np.array([x / test_sum for x in test])

        if self.apply_bayes:
            print(f'Training prior is {self.train_prior}')
            print(f'Test prior is {self.test_prior}')

    def split(self, x, y):
        """
        - Split data into train, val, test (70%, 20%, 10%)
        - Compute and save priors
        - Add the EvaluationCallback to this.callbacks.
            This callback evaluates the test set after each epoch, and adds the current accuracy loss and f1 to tensorboard.
            The results are of course not used for training, but rather to show progress on the test set during training
        """
        # Split data
        x_train_val, self.x_test, y_train_val, self.y_test = train_test_split(x, y, test_size=0.1, random_state=1337)

        # Rebalance if desired
        if self.rebalance_test is not None:
            write_log(f'Rebalancing test set to {self.rebalance_test}')
            self.x_test, self.y_test = self.__rebalance(self.x_test, self.y_test, self.rebalance_test)

        # Compute priors
        self.compute_priors(y_train_val, self.y_test)
        # Add Evaluation callback
        self.callbacks.append(EvaluationCallback(self))

        # Split train_val into train and val
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train_val, y_train_val, test_size=.2 / 0.9, random_state=1338)
        if self.rebalance_train_val is not None:
            write_log(f'Rebalancing train and validation set to {self.rebalance_train_val}')
            self.x_train, self.y_train = self.__rebalance(self.x_train, self.y_train, self.rebalance_train_val)
            self.x_val, self.y_val = self.__rebalance(self.x_val, self.y_val, self.rebalance_train_val)

    @staticmethod
    def __rebalance(x: pd.DataFrame, y: pd.DataFrame, target_distribution: []):
        """
        Rebalance a dataset by downsampling the majority classes
        :param x: The train data
        :param y: The labels
        :param target_distribution: The desired new distribution. Should be of length len(y) and sum to 1
        :return:
        """
        previous_size = len(y)
        target_distribution = np.array(target_distribution) * 100
        current_counts = np.array(y.value_counts().array)
        one_percents = current_counts / target_distribution
        count_of_one_percent = min(one_percents)
        new_counts = np.array(target_distribution * count_of_one_percent).astype(int)
        rebalanced_indices = []
        number_of_classes = len(target_distribution)
        if number_of_classes == 2:
            for class_value in [0, 1]:
                indices = y.loc[y == class_value].index.array
                sampled_indices = np.random.choice(indices, min(new_counts[class_value], len(indices)), replace=False)
                rebalanced_indices.extend(sampled_indices)
        else:
            for label, count in zip(y.columns.values, new_counts):
                all_indices = y.loc[y[label] == 1].index
                sample = np.random.choice(all_indices.array, min(count, len(all_indices)), replace=False)
                rebalanced_indices.extend(sample)
        x = x.loc[rebalanced_indices]
        y = y.loc[rebalanced_indices]

        new_size = len(y)
        drop_count = previous_size - new_size
        write_log(f'Rebalancing dropped {drop_count} of the {previous_size} rows ({drop_count / previous_size * 100}%)')
        return x, y

    @abc.abstractmethod
    def get_net(self) -> training.Model:
        """
        Returns a base network, must be extended by specific implementations
        Final layer returns RELU activation after Dense(8)
        :rtype: training.Model
        """
        weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1)
        net = Sequential()
        net.add(Dense(256, input_dim=self.x_train.shape[1], kernel_initializer=weight_initializer))
        net.add(BatchNormalization())
        # net.add(Dropout(0.3))
        net.add(LeakyReLU())
        net.add(Dense(512))
        net.add(Dropout(0.3))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        # net.add(Dense(1024))
        # net.add(Dropout(0.2))
        # net.add(BatchNormalization())
        # net.add(LeakyReLU())
        # net.add(Dense(512))
        # net.add(Dropout(0.2))
        # net.add(BatchNormalization())
        # net.add(LeakyReLU())
        net.add(Dense(256))
        net.add(BatchNormalization())
        net.add(Dropout(0.3))
        net.add(LeakyReLU())
        net.add(Dense(128))
        # net.add(Dropout(0.4))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(64))
        net.add(BatchNormalization())
        net.add(Dropout(0.4))
        net.add(LeakyReLU())
        net.add(Dense(32))
        # net.add(Dropout(0.3))
        net.add(BatchNormalization())
        net.add(LeakyReLU())
        net.add(Dense(16))
        net.add(BatchNormalization())
        net.add(Dropout(0.2))
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
        - Use verbose output to show accuracy, loss and f1 during training
        :return the trained network and the history
        """
        write_log('Loading data')
        self.load_data()

        net = self.compile_net()
        write_log('Training network')

        history = net.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            epochs=self.num_epochs,
            verbose=1,
            callbacks=self.callbacks,
        )
        write_log('Training finished')
        # self.__show_history(history)
        return net, history

    def __show_history(self, history):
        """
        Plot accuracy per epoch using pyplot
        @param history:
        """
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def _apply_bayes(self, y_pred):
        """
        Apply bayes: posterior = prediction * prior_test / prior_train
        @param y_pred: The prediction
        @return: Posterior
        """
        test_prior = np.array(self.test_prior)
        train_prior = np.array(self.train_prior)
        posterior = np.array([test_prior * y / train_prior for y in y_pred])
        return posterior

    def test_binary(self, net: training.Model, verbose=True):
        """
        Test the networks performance on binary classification
        :param net the trained network
        """
        # Labels to binary
        if isinstance(self.y_test, pd.DataFrame):
            y_true = self.y_test['4top'].to_numpy()
        else:
            y_true = self.y_test.to_numpy()

        y_pred = net.predict(self.x_test)
        # Convert single value probs to 2 value probs
        if y_pred.shape[1] == 1:
            y_pred = np.array([[y[0], 1 - y[0]] for y in y_pred])
        # Apply bayes
        if self.apply_bayes:
            y_pred = self._apply_bayes(y_pred)

        # Select the class with the highest probability
        classes = np.argmax(y_pred, axis=1)
        # Convert to binary
        y_pred = [int(y == 0) for y in classes]

        pred_true = sum(y_pred)
        pred_false = len(y_pred) - pred_true
        write_log(f'Num predicted true: {pred_true}\n')
        write_log(f'Num predicted false: {pred_false}\n')

        loss = log_loss(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        if verbose:
            write_log(f'Test loss, f1, and accuracy are {loss}, {f1}, {accuracy}')
        return loss, f1, accuracy

    def test(self, net: training.Model, verbose=True):
        """
        Test the network, compute accuracy, loss, and f1
        :param net the trained network
        """
        # Get numpy data
        y_true = self.y_test.to_numpy()
        y_pred = net.predict(self.x_test)
        # Apply bayes
        if self.apply_bayes:
            y_pred = self._apply_bayes(y_pred)

        # Compute categorical cross entropy
        loss = log_loss(y_true, y_pred)
        # Convert probs to hard classes: select class with max probability
        y_pred = np.argmax(y_pred, axis=1)
        # One-hot encode
        y_pred = np.eye(5)[y_pred]
        # Compute accuracy and f1
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        if verbose:
            write_log(f'Test loss, f1, and accuracy are {loss}, {f1}, {accuracy}')
        return loss, f1, accuracy

    def categorical_crossentropy(self):
        """
        Categorical cross entropy loss function. Use weighted version if self.class_weights is provided
        @param self:
        @return:
        """
        if self.class_weights is not None:
            return self.weighted_categorical_crossentropy(self.class_weights)
        else:
            return tf.keras.losses.CategoricalCrossentropy()

    @staticmethod
    def f1(y_true, y_pred):
        """
        F1 loss function
        @param y_true:
        @param y_pred:
        @return:
        """

        def recall(y_true, y_pred):
            true_positives = K.mean(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.mean(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = K.mean(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.mean(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss
