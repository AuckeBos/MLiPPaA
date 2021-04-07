import argparse

from Element2.BaseClassification import BaseClassifier
from Element2.BinaryClassification import BinaryClassifier
from Element2.DataLoader import DataLoader
from Element2.MultiClassification import MultiClassifier
from Element2.RecurrentClassification import RecurrentClassifier
from tensorflow.keras.models import load_model
import csv
import pandas as pd

def read():
    parser = argparse.ArgumentParser(
        description='Load the best version for each model; test them on a test dataset; save predictions to csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--data-file', type=str, required=True, help='The datafile containing the test data')
    args = parser.parse_args()
    return args.data_file


def run(data_file):
    binary_classifier_file = './best_binary_classifier.h5'
    multi_classifier_file = './best_multiclass_classifier.h5'
    recurrent_classifier_file = './best_recurrent_classifier.h5'

    binary_classifier = BinaryClassifier()
    multi_classifier = MultiClassifier()
    recurrent_classifier = RecurrentClassifier()

    binary_net = load_model(binary_classifier_file, custom_objects={'f1': BaseClassifier.f1, 'loss': binary_classifier.loss()})
    multi_net = load_model(multi_classifier_file, custom_objects={'f1': BaseClassifier.f1, 'loss': multi_classifier.loss()})
    recurrent_net = load_model(recurrent_classifier_file, custom_objects={'f1': BaseClassifier.f1, 'loss': recurrent_classifier.loss()})

    classifier_data = [
        [binary_classifier, 'binary_predictions.csv', binary_net],
        [multi_classifier, 'multi_predictions.csv', multi_net],
        [recurrent_classifier, 'recurrent_predictions.csv', recurrent_net],
    ]
    # todo: ids seem not to be unique?
    ids = pd.read_csv(data_file, delimiter=';', usecols=[0], names=['EventID'])['EventID'].tolist()
    for (classifier, filename, net) in classifier_data:
        x, _ = classifier.load_data(data_file)
        predictions = classifier.predict(net, x)
        with open(filename, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            for (prediction, id) in zip(predictions, ids):
                writer.writerow([int(id), prediction])


if __name__ == '__main__':
    data_file = read()
    run(data_file)
