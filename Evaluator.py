import csv
import itertools
import json

from BaseClassification import BaseClassifier
from RecurrentClassification import RecurrentClassifier
from helpers import write_log


class Evaluator:
    """
    Simple static class that provides evaluation functionality
    """

    @staticmethod
    def evaluate(classifier: BaseClassifier):
        """
        Evaluate a classifier by training and testing it
        :param classifier:
        """
        net, history = classifier.train()
        classifier.test(net)
        return net

    @staticmethod
    def compare_design_choices():
        """
        Compare a combination of 8 different design choices. Save training history in json format
        """
        crossentropy_weights = [10, 1, 1, 1, 1]
        rebalance_to = [0.04] + ([(1 - 0.04) / 4] * 4)
        apply_bayes_options = [True, False]
        rebalance_testset_options = [True, False]
        rebalance_trainvalidation_options = [True, False]
        weighted_loss_options = [True, False]
        comparison = []
        for i, (apply_bayes, rebalance_test, rebalance_trainval, weighted_loss) in enumerate(list(itertools.product(apply_bayes_options, rebalance_testset_options, rebalance_trainvalidation_options, weighted_loss_options))):
            classifier = RecurrentClassifier()
            classifier.apply_bayes = apply_bayes
            if rebalance_test:
                classifier.rebalance_test = rebalance_to
            else:
                classifier.rebalance_test = None
            if rebalance_trainval:
                classifier.rebalance_train_val = rebalance_to
            else:
                classifier.rebalance_train_val = None
            if weighted_loss:
                classifier.class_weights = crossentropy_weights
            else:
                classifier.class_weights = None
            net, history = classifier.train()
            history = history.history
            history['lr'] = [float(lr) for lr in history['lr']]
            comparison.append({
                'apply_bayes': apply_bayes,
                'rebalance_test': rebalance_test,
                'rebalance_trainval': rebalance_trainval,
                'weighted_loss': weighted_loss,
                'history': history
            })
            write_log(f'Done training {i}')
        with open('comparison.json', 'w+') as fp:
            json.dump(comparison, fp)

    @staticmethod
    def evaluate_comparison(file):
        """
        After compare_design_choices has been ran, we have a comparison in json format. Convert to csv by saving the best values for each hyper parameter combination
        @param file: The json comparison file
        """
        with open(file) as json_file:
            content = json.load(json_file)
            for i, experiment in enumerate(content):
                experiment['val_loss'] = min(experiment['history']['val_loss'])
                experiment['val_accuracy'] = max(experiment['history']['val_accuracy'])
                experiment['val_f1'] = max(experiment['history']['val_f1'])

                experiment['test_loss'] = min(experiment['history']['test_loss'])
                experiment['test_accuracy'] = max(experiment['history']['test_accuracy'])
                experiment['test_f1'] = max(experiment['history']['test_f1'])

                experiment['epochs'] = len(experiment['history']['val_loss'])
                experiment['learningrate'] = min(experiment['history']['lr'])

                content[i] = experiment
        with open('comparison.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            headers = ['apply_bayes', 'rebalance_test', 'rebalance_trainval', 'weighted_loss', 'val_loss', 'val_accuracy', 'val_f1', 'test_loss', 'test_accuracy', 'test_f1', 'epochs', 'learningrate']
            writer.writerow(headers)
            for experiment in content:
                writer.writerow([experiment[header] for header in headers])
