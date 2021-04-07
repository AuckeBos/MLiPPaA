from tensorflow.keras.layers import Dense, LSTM, Concatenate, Masking, LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Element2.BaseClassification import BaseClassifier
from Element2.DataLoader import DataLoader
from Element2.helpers import to_ndarray


class RecurrentClassifier(BaseClassifier):
    """
    Our more sophisticated network for multi-class classification: An RNN using LSTMs
    """

    # The columns that hold the event level attributes
    event_level_columns = ['MET', 'METphi', 'VectorCount']

    # The column that holds this lists of vectors
    vector_column = 'Vectors'

    batch_size = 64
    num_epochs = 150

    def get_net(self):
        """
        Get the network. Use a recurrent neural network with variable input length.
        The network has two inputs
        - The first input has shape (3,): met, metphi, object count
        - The second input has shape (19, 11): 19 vectors of length 11. Non existing objects are padded '-100', we mask these out.

        The 11 values vor each measured object are:
            - The first 6 are the one-hot encoding of the object type
            - The last 5 are  Charge, E, Pt, Eta, Phi
        """
        # Load vector shape (19, 11) By the value of the first row
        vectors_shape = self.x_train[1][0].shape
        # Event data has 3 inputs: (3,)

        event_data = Input(shape=(len(self.event_level_columns),))
        object_data = Input(vectors_shape)
        inputs = [event_data, object_data]

        # Mask object data
        object_data = Masking(mask_value=DataLoader.padding_mask, input_shape=vectors_shape)(object_data)
        # LSTM for object data
        object_data = LSTM(30, return_sequences=True, dropout=0.5)(object_data)
        object_data = LSTM(40, return_sequences=True, dropout=0.5)(object_data)
        object_data = LSTM(20, return_sequences=True, dropout=0.5)(object_data)
        object_data = LSTM(10, return_sequences=False, dropout=0.5)(object_data)

        # Concatenate LSTM output with event data
        merged = Concatenate()([event_data, object_data])

        # Down sample to 5, final softmax activation
        merged = Dense(10)(merged)
        merged = LeakyReLU()(merged)
        outputs = Dense(5)(merged)
        outputs = Softmax()(outputs)
        net = Model(inputs, outputs)

        return net

    def load_data(self, data_file: str = None):
        """
        Load the data using the dataloader
        """
        data_loader = DataLoader()
        data_loader.variable_input_length = True
        x, y = data_loader.load_data(data_file)
        # Save one-hot to string mapping
        self.predictions_to_labels = data_loader.predictions_to_labels
        self.split(x, y)

        # Now convert the training sets such that they consist of two inputs of size (#Points, 3) and (#Points, 1)
        self.x_train = self.restructure_input(self.x_train)
        self.x_val = self.restructure_input(self.x_val)
        self.x_test = self.restructure_input(self.x_test)

        # Return complete unsplitted set
        return self.restructure_input(x), y

    def restructure_input(self, x):
        """
        The nwtork requires a restructure of the traning data such that they consist of two inputs of size (#Points, 3) and (#Points, 1). Do so here
        @param x:  The original data
        @return:  The restructured data
        """
        return [x[self.event_level_columns], to_ndarray(x[self.vector_column].values)]

    def compile_net(self):
        """
        Use adam optimizer, categorical cross entropy
        :return: the net
        """
        optimizer = Adam(learning_rate=.0001)
        net = self.get_net()
        net.compile(loss=self.categorical_crossentropy(), optimizer=optimizer, metrics=['accuracy', self.f1])
        return net
