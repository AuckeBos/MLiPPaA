import numpy as np
from sklearn.metrics import log_loss, f1_score, accuracy_score
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import training

from Element2.BaseClassification import BaseClassifier
from Element2.DataLoader import DataLoader
from Element2.helpers import write_log


class MultiClassifier(BaseClassifier):
    """
    Our classifier used for multi-class classification
    """
    batch_size = 64
    num_epochs = 150

    def get_net(self):
        """
        Get the network. Use the base network, append Dense(5) with softmax
        """
        net = super().get_net()
        net.add(Dense(5))
        net.add(Softmax())
        return net

    def load_data(self, data_file: str = None):
        """
        Load the data using the dataloader
        """
        data_loader = DataLoader()
        x, y = data_loader.load_data(data_file)
        # Save one-hot to string mapping
        self.predictions_to_labels = data_loader.predictions_to_labels
        self.split(x, y)

        # Return complete unsplitted set
        return x, y


    def compile_net(self):
        """
        Use adam optimizer, categorical cross entropy
        :return: the net
        """
        optimizer = Adam(learning_rate=.0001)
        net = self.get_net()
        net.compile(loss=self.loss(), optimizer=optimizer, metrics=['accuracy', self.f1])
        return net

    def test_binary(self, net: training.Model, verbose=True):
        """
        Test the networks performance on binary classification
        :param net the trained network
        """
        y_true = self.y_test.to_numpy()

        y_pred = net.predict(self.x_test)
        # Apply bayes
        if self.apply_bayes:
            y_pred = self._apply_bayes(y_pred)

        # Select the class with the highest probability
        classes = np.argmax(y_pred, axis=1)
        # Convert to binary
        y_pred = [int(y == 0) for y in classes]

        loss = log_loss(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        if verbose:
            write_log(f'Test loss, f1, and accuracy are {loss}, {f1}, {accuracy}')
        return loss, f1, accuracy
