from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam

from BaseClassification import BaseClassifier
from DataLoader import DataLoader


class MultiClassifier(BaseClassifier):
    batch_size = 64
    num_epochs = 150

    def get_net(self):
        """
         Get the network. Use the base network, append Dens(5) with softmax
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
        net = self.get_net()
        net.compile(loss=self.categorical_crossentropy(), optimizer=optimizer, metrics=['accuracy', self.f1])
        return net
