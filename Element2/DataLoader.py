from typing import List
import numpy as np
import pandas as pd
from numpy.core._multiarray_umath import ndarray
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """
    Loads and preprocesses the data
    """

    # File that holds the data
    data_file = './data/TrainingValidationData_200k_shuffle.csv'

    # Columns holding event data
    event_columns = np.array(['EventID', 'ProcessID', 'EventWeight', 'MET', 'METphi'])

    # Use this encoding for manual particle one-hot encoding, to ensure that all v-vectors are of equal length
    manual_particle_encoding = {
        'b': [1, 0, 0, 0, 0, 0],
        'e': [0, 1, 0, 0, 0, 0],
        'g': [0, 0, 1, 0, 0, 0],
        'j': [0, 0, 0, 1, 0, 0],
        'm': [0, 0, 0, 0, 1, 0],
        'p': [0, 0, 0, 0, 0, 1],
    }

    # The object columns are all particle types, charge, 4-vector values (size 11)
    object_columns = [f'Particle_{p}' for p in manual_particle_encoding.keys()] + ['Charge', 'E', 'Pt', 'Eta', 'Phi']

    # The labels are stored in this column
    label_column = 'ProcessID'

    # The data and labels DFs
    x: pd.DataFrame = None
    y: pd.DataFrame = None

    # The name of the column that holds the labels
    label_column: str

    # The names of the columns that hold the data about the objects
    object_columns: List[str]

    # The names of the columns that hold the data about the events
    event_columns: ndarray

    # The file that holds the data
    data_file: str

    # Valid options are 'multi', 'binary'
    classification_type: str = 'multi'

    # Only normalize if set to true
    normalize_data: bool = True

    # If true, load data as numpy array instead of dataframe, with a variable number of objects
    # Used when in a
    variable_input_length: bool = False

    # If we have variable input length, we pad with '-100'
    padding_mask = -100

    # Used to map predictions \in [0,5] back to string labels. Is a simple array of length 5 that maps index to string
    predictions_to_labels:  List[str] = None

    def set_binary_classification(self):
        """
        Set DataLoader to use binary classification, this alters the value of 'y' into binary labels
        """
        self.classification_type = 'binary'

    @staticmethod
    def __4_vector_to_array(string):
        """
        The 4-vectors are initially loaded as string. In this function:
        - Split the string into an array
        - Add the charge of the particle as int (-1, 0, 1), and remove the '-' or '+' from the particle name
        :param string: The 4-vector as string
        :return: The 4-vector as array
        """
        # Return empty array for the NaN values, this is the zero padding
        if type(string) == float:
            return []

        splitted = str(string).split(',')
        particle = splitted[0]
        charge = 0
        charge_string = particle[-1]
        # If charge is provided
        if charge_string in ['-', '+']:
            # Remove the char
            splitted[0] = splitted[0][:-1]
            # Save the charge
            charge = -1 if charge_string == '-' else 1
        # Include the charge in the array
        splitted.insert(1, charge)

        # as_array format: one-hot-encoded particle, charge, remaining 4-vector values (size (11,))
        as_array = [float(x) for x in DataLoader.manual_particle_encoding[splitted[0]] + splitted[1:]]
        return as_array

    def __explode_4_vectors(self):
        """
        self.x is the dataframe with all training columns. In this function, we split the 4-vectors.
        The 4-vectors are initially loaded as comma-separated string, each vector in its own columns.
        Now for each object, split the string into separate columns. Also one-hot encode the object types.
        """
        # Get all columns names that start with Object
        object_columns = [col for col in self.x if col.startswith('Object')]
        # Will be used to store vectors in case self.variable_input_length=True
        vectors = pd.DataFrame(columns=[f'Vector{i + 1}' for i in range(len(object_columns))])
        for i, column_name in enumerate(object_columns):
            obj_num = i + 1
            # Convert the string into an array, will also include a 'Charge' column
            column_split = self.x[column_name].map(self.__4_vector_to_array)
            # Now add the values at named columns. self.object_columns shows the names of these columns
            column_names = [f'{col}_{obj_num}' for col in self.object_columns]
            # Create dataframe with these columns
            new_columns = pd.DataFrame(column_split.to_list(), columns=column_names)
            # Delete old columns 'Object1' columns, add the vector columns
            del self.x[column_name]
            # If variable input length, save vectors in one vector column. These vector columns will be merged into 1 column later
            if self.variable_input_length:
                vectors[f'Vector{obj_num}'] = new_columns.values.tolist()
            else:
                self.x[new_columns.columns] = new_columns
        # Merge all column vectors into one column of vector lists
        if self.variable_input_length:
            # Pad non existing vectors to '-100'
            zero_padded_vectors = []
            for row in vectors.values.tolist():
                vectors_for_row = []
                for vector in row:
                    if np.isnan(vector).any():
                        vector = [DataLoader.padding_mask] * len(vector)
                    vectors_for_row.append(vector)
                zero_padded_vectors.append(vectors_for_row)
            self.x['Vectors'] = zero_padded_vectors

    def __load_labels(self, df):
        """
        Load y, based on the classification type
        """
        # Return binary labels
        if self.classification_type == 'binary':
            labels = df[self.label_column].map(lambda process: 1 if process == '4top' else 0)
            self.predictions_to_labels = ['background', '4top']
        # Return one hot encoded labels
        elif self.classification_type == 'multi':
            labels = df[self.label_column]
            # One hot encoding
            labels = pd.get_dummies(labels)
            self.predictions_to_labels = labels.columns.tolist()
        else:
            raise Exception(f'Invalid classification type {self.classification_type}')
        self.y = labels

    def __normalize_data(self):
        """
        Normalize the data set to range [0, 1]
        """
        # Normalize all columns, except the particle names, charge, and Vector
        column_names_to_normalize = [col for col in self.x.columns if 'Object' not in col and 'Charge' not in col and 'Vector' not in col]
        values = self.x[column_names_to_normalize].values
        scaled = MinMaxScaler().fit_transform(values)
        df = pd.DataFrame(scaled, columns=column_names_to_normalize, index=self.x.index)
        self.x[column_names_to_normalize] = df

    def load_data(self, data_file=None, has_labels=True):
        """
        Load the data from csv

        @param data_file: If provided, use this file to load data from. If not provided, use self.data_file
        @param has_labels: If true, assume the data contains labels. In this case we will load self.y too, and exclude the label column in self.x
        @return: self.x, self.y. self.y is None if has_labels=False
        """
        if data_file is None:
            data_file = self.data_file

        # Add columns for 4-vectors
        object_columns = np.array([f'Object{i}' for i in range(1, 25)])
        all_columns = np.concatenate((self.event_columns, object_columns.flatten()))
        # Note: 4-vector data is ',' separated, thus each vector is saved as string in 1 column
        data = pd.read_csv(data_file, names=all_columns, sep=';')
        # Created 25 object columns, might be too many. Drop columns with ALL NaNs
        data.dropna(axis=1, how='all', inplace=True)

        # Add column that holds the amount of non-null vectors
        max_nr_of_vectors = len([col for col in data if col.startswith('Object')])
        data['VectorCount'] = max_nr_of_vectors - data.isnull().sum(axis=1)
        # Drop columns that we do not want to train on
        self.x = data.drop(columns=['EventID', 'EventWeight'])
        # If labels are provided, drop them from training
        if has_labels:
            self.x.drop(columns=[self.label_column], inplace=True)

        self.__explode_4_vectors()

        # Fill nans (zero padding)
        self.x.fillna(0, inplace=True)
        # If we must normalize, do so
        if self.normalize_data:
            self.__normalize_data()
        if has_labels:
            self.__load_labels(data)
        return self.x, self.y
