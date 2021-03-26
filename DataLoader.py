from typing import List

import numpy as np
import pandas as pd
from numpy.core._multiarray_umath import ndarray
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
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

    def __init__(self):
        """
        Set config values of the data
        """
        self.data_file = './data/TrainingValidationData_200k_shuffle.csv'
        self.event_columns = np.array(['EventID', 'ProcessID', 'EventWeight', 'MET', 'METphi'])
        self.object_columns = [f'Object', f'Charge', f'E', f'Pt', f'Eta', f'Phi']
        self.label_column = 'ProcessID'

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
        array = str(string).split(',')
        particle = array[0]
        charge = 0
        charge_string = particle[-1]
        # If charge is provided
        if charge_string in ['-', '+']:
            # Remove the char
            array[0] = array[0][:-1]
            # Save the charge
            charge = -1 if charge_string == '-' else 1
        # Include the charge in the array
        array.insert(1, charge)

        # Convert the floats to actual floats
        array = [float(x) if i > 1 else x for i, x in enumerate(array)]
        return array

    def __explode_4_vectors(self):
        """
        self.x is the dataframe with all training columns. In this function, we split the 4-vectors.
        The 4-vectors are initially loaded as comma-separated string, each vector in its own column.
        Now for each object, split the string into separate columns. Also one-hot encode the object types.
        """
        # Get all columns names that start with Object
        object_columns = [col for col in self.x if col.startswith('Object')]
        for i, column_name in enumerate(object_columns):
            obj_num = i + 1
            # Convert the string into an array, will also include a 'Charge' column
            column_split = self.x[column_name].map(self.__4_vector_to_array)
            # Now add the values at named columns. self.object_columns shows the names of these columns
            column_names = [f'{col}{obj_num}' for col in self.object_columns]
            columns = pd.DataFrame(column_split.to_list())
            # Delete old columns 'Object1', which contained the full not-split string
            del (self.x[column_name])
            # Add the new columns
            self.x[column_names] = columns
            # One hot encode the values in the new Object{obj_num} column, which now holds the 1-letter object type
            one_hot_obj_type = pd.get_dummies(self.x[column_name], prefix=column_name, drop_first=True)
            self.x[one_hot_obj_type.columns] = one_hot_obj_type
            # One-hot encoded columns are added, thus remove the old 'Object{obj_num} column
            del (self.x[column_name])

    def __load_labels(self, df):
        """
        Load y, based on the classification type
        """
        # Return binary labels
        if self.classification_type == 'binary':
            labels = df[self.label_column].map(lambda process: 1 if process == '4top' else 0)
        # Return one hot encoded labels
        elif self.classification_type == 'multi':
            labels = df[self.label_column]
            # One hot encoding
            labels = pd.get_dummies(labels)
        else:
            raise Exception(f'Invalid classification type {self.classification_type}')
        self.y = labels

    def __normalize_data(self):
        """
        Normalize the data set to range [0, 1]
        """
        column_names_to_normalize = [col for col in self.x.columns if 'Object' not in col and 'Charge' not in col]
        values = self.x[column_names_to_normalize].values
        scaled = MinMaxScaler().fit_transform(values)
        df = pd.DataFrame(scaled, columns=column_names_to_normalize, index=self.x.index)
        self.x[column_names_to_normalize] = df

    def load_data(self):
        """
        Load the data from csv
        :return: self.x and self.y dataframes
        """
        # Add columns for 4-vectors
        object_columns = np.array([f'Object{i}' for i in range(1, 25)])
        all_columns = np.concatenate((self.event_columns, object_columns.flatten()))
        # Note: 4-vector data is ',' separated, thus each vector is saved as string in 1 column
        data = pd.read_csv(self.data_file, names=all_columns, sep=';')
        # Created 25 object columns, might be too many. Drop columns with ALL NaNs
        data.dropna(axis=1, how='all', inplace=True)
        # Drop columns that we do not want to train on
        self.x = data.drop(columns=['EventID', 'ProcessID', 'EventWeight'])
        self.__explode_4_vectors()
        # Fill nans (zero padding)
        self.x.fillna(0, inplace=True)
        # If we must normalize, do so
        if self.normalize_data:
            self.__normalize_data()
        self.__load_labels(data)
        return self.x, self.y
