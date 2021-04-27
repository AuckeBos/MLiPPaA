import argparse

import Element2.Evaluator
from Element2.BaseClassification import BaseClassifier
from Element2.BinaryClassification import BinaryClassifier
from Element2.DataLoader import DataLoader
from Element2.MultiClassification import MultiClassifier
from Element2.RecurrentClassification import RecurrentClassifier
from tensorflow.keras.models import load_model
import csv
import pandas as pd
import numpy as np

# As computed by the training data distribution (RebalanceTrainVal=False)
multi_train_prior = np.array([.5, .125, .125, .125, .125])
binary_train_prior = np.array([.5, .5])

multi_test_prior = np.array([.04, .02, .19, .51, .24])
binary_test_prior = np.array([.96, .04])


def read():
    parser = argparse.ArgumentParser(
        description='Load a model, test them on a test dataset; save predictions to csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--data-file', type=str, required=False, default='../Element2/data/ExamData.csv', help='The datafile containing the test data')
    parser.add_argument('-t', '--classification-type', type=str, required=True, choices=['binary', 'multi'], help='Classification type: multi label or binary')
    parser.add_argument('-m', '--model', type=str, required=True, choices=['binary', 'multi', 'recurrent'], help='Classification model: BinaryClassifier, MultiClassifier, or RecurrentClassifier')
    parser.add_argument('-h5', '--h5', type=str, required=True, help='The h5 file of the saved model')
    parser.add_argument('-b', '--bayes', type=str, required=True, choices=['True', 'False'], help='Apply bayes to the prediction outputs?')
    args = parser.parse_args()
    return args.data_file, args.classification_type, args.model, args.h5, bool(args.bayes)


def run(data_file, classification_type, model_type, h5, apply_bayes):
    classifier = Element2.Evaluator.Evaluator.parse_classifier_type(model_type)
    classifier.apply_bayes = apply_bayes
    if model_type == 'binary':
        classifier.train_prior = binary_train_prior
        classifier.test_prior = binary_test_prior
    else:  # Multi or recurrent
        classifier.train_prior = multi_train_prior
        classifier.test_prior = multi_test_prior
    net = load_model(h5, custom_objects={'f1': BaseClassifier.f1, 'loss': classifier.loss()})

    # Use manual label mapping for multi classifier:
    predictions_to_labels = ['4top', 'ttbar', 'ttbarHiggs', 'ttbarW', 'ttbarZ']
    # Define the number of objects per row. Needed because we need to have the exact same input shape as during training, otherwise
    # The network won't be able to predict. Note that this does not decrease performance, since the network will mask them out
    objects_per_row = 19
    ids = pd.read_csv(data_file, delimiter=';', usecols=[0], names=['EventID'])['EventID'].tolist()
    x, _ = classifier.load_data(data_file, False, objects_per_row)
    predictions = classifier.predict(net, x)
    with open('predictions.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for (prediction, id) in zip(predictions, ids):
            # Prefix with labels
            if classification_type == 'binary':  # prediction[0] must be the probability of 4-top
                prediction = [f'4top={prediction[0]}']
            else:  # multi: prediction is array of probs
                prediction = [f'{label}={value}' for (label, value) in zip(predictions_to_labels, prediction)]
            writer.writerow([int(id)] + prediction)


if __name__ == '__main__':
    run(*read())
