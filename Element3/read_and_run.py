import argparse
import csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import Element2.Evaluator
from Element2.BaseClassification import BaseClassifier

# As computed by the training data distribution (RebalanceTrainVal=False)
multi_train_prior = np.array([.5, .125, .125, .125, .125])
binary_train_prior = np.array([.5, .5])

multi_test_prior = np.array([.04, .02, .19, .51, .24])
binary_test_prior = np.array([.96, .04])


def read():
    """
    Read command line arguments for the script:
    - --data-file: The data file with the data to test. If not provided, use ExamData.csv in /data
    - --classification-type: Classify binary or multiclass
    - --model: Which type of model to use: The BinaryClassifier, MultiClassifier, or RecurrentClassifier
    - --h5: The h5 file of the pretrained model, should match with --model
    - --bayes: Apply bayes on the predictions
    @return:
    """
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


def run(data_file: str, classification_type: str, model_type: str, h5: str, apply_bayes: bool):
    """
    After commandline args have been read, run the model:
    - Load the classifier
    - Load the data
    - Predict the data
    - Generate csv in the desired format (predictions.csv)
    @param data_file: The file that contains the testset
    @param classification_type: The type of classification: binary or multi
    @param model_type: The classifier type: binary, multi, recurrent
    @param h5:  The h5 file of the trained model
    @param apply_bayes:  Bool that indicates whether to apply bayes on the predictions
    """
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
