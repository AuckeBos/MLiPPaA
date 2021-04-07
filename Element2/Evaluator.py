import csv
import itertools
import json
from tensorflow.keras.models import load_model
from Element2.BaseClassification import BaseClassifier
from Element2.BinaryClassification import BinaryClassifier
from Element2.MultiClassification import MultiClassifier
from Element2.RecurrentClassification import RecurrentClassifier
from Element2.helpers import write_log


class Evaluator:
    """
    Static class that provides evaluation functionality
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
    def compare_design_choices(classifier_type: str, dir: str):
        """
        Compare a combination of 16 different design choices. Save training history in json format
        @param classifier_type: The classifier type, must be 'binary', 'multi' or 'recurrent'
        @param dir:  The dir to save the best performing models, and the comparison to
        """
        # Configure the distributions and weights, depending on the classifier type
        if classifier_type == 'binary':
            rebalance_to = [1 - 0.04, 0.04]
            loss_weights = [1, 10]
        else:
            loss_weights = [10, 1, 1, 1, 1]
            rebalance_to = [0.04] + ([(1 - 0.04) / 4] * 4)

        # Load all configurations
        apply_bayes_options = [True, False]
        rebalance_testset_options = [True, False]
        rebalance_trainvalidation_options = [True, False]
        weighted_loss_options = [True, False]
        comparison = []
        for i, (apply_bayes, rebalance_test, rebalance_trainval, weighted_loss) in enumerate(list(itertools.product(apply_bayes_options, rebalance_testset_options, rebalance_trainvalidation_options, weighted_loss_options))):
            # For each configuration, train a model, save its best performing instance
            if classifier_type == 'binary':
                classifier = BinaryClassifier()
            elif classifier_type == 'multi':
                classifier = MultiClassifier()
            elif classifier_type == 'recurrent':
                classifier = RecurrentClassifier()
            else:
                raise Exception('No classifier exists for ' + classifier_type)

            classifier.save_to = f'{dir}/{i}/'
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
                classifier.class_weights = loss_weights
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
        # Save resulting data as json
        with open(f'{dir}/comparison.json', 'w+') as fp:
            json.dump(comparison, fp)

    @staticmethod
    def comparison_to_csv(file):
        """
        After compare_design_choices has been ran, we have a comparison in json format. Convert to csv by saving the best values for each hyper parameter combination
        @param file: The json comparison file
        """
        with open(file) as json_file:
            content = json.load(json_file)
            # For each configuration, save the relevant metrics to a dictionary
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
        # Write the dictionary to csv
        with open('comparison.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            headers = ['apply_bayes', 'rebalance_test', 'rebalance_trainval', 'weighted_loss', 'val_loss', 'val_accuracy', 'val_f1', 'test_loss', 'test_accuracy', 'test_f1', 'epochs', 'learningrate']
            writer.writerow(headers)
            for experiment in content:
                writer.writerow([experiment[header] for header in headers])

    @staticmethod
    def compare_binary_performance(dir_binary_classifier: str, dir_multi_classifier: str):
        """
        Compare performance of the binary classifier with the multi classifier. Assumes models are yet trained and saved using compare_design_choices.
        @param dir_binary_classifier: Dir where the binary classifier models are saved
        @param dir_multi_classifier: Dir where the multi classifier models are saved
        """
        rebalanced_distribution = [1 - 0.04, 0.04]
        # Get the sets
        classifier = BinaryClassifier()
        classifier.load_data()
        x_not_rebalanced, y_not_rebalanced = classifier.x_test, classifier.y_test

        classifier = BinaryClassifier()
        classifier.rebalance_test = [1 - 0.04, 0.04]
        classifier.load_data()
        x_rebalanced, y_rebalanced = classifier.x_test, classifier.y_test
        # Use these configurations again, such that we enumerate over the same indices as in compare_design_choices
        apply_bayes_options = [True, False]
        rebalance_testset_options = [True, False]
        rebalance_trainvalidation_options = [True, False]
        weighted_loss_options = [True, False]
        comparison = []
        binary_classifier = BinaryClassifier()
        multi_classifier = MultiClassifier()
        for i, (apply_bayes, _, _, _) in enumerate(list(itertools.product(apply_bayes_options, rebalance_testset_options, rebalance_trainvalidation_options, weighted_loss_options))):
            # For each configuration, test both networks on both sets
            current_evaluation = {}
            binary_net = load_model(f'{dir_binary_classifier}/{i}', custom_objects={'f1': BaseClassifier.f1, 'loss': binary_classifier.loss()})
            multi_net = load_model(f'{dir_multi_classifier}/{i}', custom_objects={'f1': BaseClassifier.f1, 'loss': multi_classifier.loss()})

            # Evaluate the not rebalanced set
            binary_classifier.x_test = x_not_rebalanced
            binary_classifier.y_test = y_not_rebalanced
            if apply_bayes:
                binary_classifier.apply_bayes = apply_bayes
                binary_classifier

            multi_classifier.x_test = x_not_rebalanced
            multi_classifier.y_test = y_not_rebalanced

            binary_loss, binary_f1, binary_accuracy = binary_classifier.test(binary_net, False)
            multi_loss, multi_f1, multi_accuracy = multi_classifier.test_binary(multi_net, False)
            current_evaluation['not_rebalanced'] = {
                'binary_loss': binary_loss,
                'binary_f1': binary_f1,
                'binary_accuracy': binary_accuracy,
                'multi_loss': multi_loss,
                'multi_f1': multi_f1,
                'multi_accuracy': multi_accuracy,
            }

            # Evaluated the rebalanced set
            binary_classifier.x_test = x_rebalanced
            binary_classifier.y_test = y_rebalanced
            multi_classifier.x_test = x_rebalanced
            multi_classifier.y_test = y_rebalanced
            binary_loss, binary_f1, binary_accuracy = binary_classifier.test(binary_net, False)
            multi_loss, multi_f1, multi_accuracy = multi_classifier.test_binary(multi_net, False)
            current_evaluation['rebalanced'] = {
                'binary_loss': binary_loss,
                'binary_f1': binary_f1,
                'binary_accuracy': binary_accuracy,
                'multi_loss': multi_loss,
                'multi_f1': multi_f1,
                'multi_accuracy': multi_accuracy,
            }
            comparison.append(current_evaluation)
            write_log(f'Done testing {i}')
        # Save results to json
        with open(f'binary_comparison.json', 'w+') as fp:
            json.dump(comparison, fp)

    @staticmethod
    def binary_comparison_to_csv(file):
        """
        After compare_binary_performance has been ran, we have a comparison in json format. Convert to csv by saving the best values for each hyper parameter combination
        @param file: The json comparison file
        """
        # Load json data
        with open(file) as json_file:
            content = json.load(json_file)
        # Use these configurations again, such that we enumerate over the same indices as in compare_binary_performance
        apply_bayes_options = [True, False]
        rebalance_testset_options = [True, False]
        rebalance_trainvalidation_options = [True, False]
        weighted_loss_options = [True, False]
        comparison = []
        for i, (apply_bayes, rebalance_test, rebalance_trainval, weighted_loss) in enumerate(list(itertools.product(apply_bayes_options, rebalance_testset_options, rebalance_trainvalidation_options, weighted_loss_options))):
            # For each configuration, save the relevant metrics to a dictionary
            experiment = {}
            experiment['apply_bayes'] = apply_bayes
            experiment['rebalance_test'] = rebalance_test
            experiment['rebalance_trainval'] = rebalance_trainval
            experiment['weighted_loss'] = weighted_loss
            data = content[i]
            rebalanced_data = data['rebalanced']
            not_rebalanced_data = data['not_rebalanced']
            experiment['rebalanced_f1_binary_classifier'] = rebalanced_data['binary_f1']
            experiment['rebalanced_f1_multi_classifier'] = rebalanced_data['multi_f1']

            experiment['not_rebalanced_f1_binary_classifier'] = not_rebalanced_data['binary_f1']
            experiment['not_rebalanced_f1_multi_classifier'] = not_rebalanced_data['multi_f1']

            experiment['rebalanced_accuracy_binary_classifier'] = rebalanced_data['binary_accuracy']
            experiment['rebalanced_accuracy_multi_classifier'] = rebalanced_data['multi_accuracy']

            experiment['not_rebalanced_accuracy_binary_classifier'] = not_rebalanced_data['binary_accuracy']
            experiment['not_rebalanced_accuracy_multi_classifier'] = not_rebalanced_data['multi_accuracy']

            comparison.append(experiment)
        # Write the dictionary to csv
        with open('binary_comparison.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            headers = ['apply_bayes', 'rebalance_test', 'rebalance_trainval', 'weighted_loss',
                       'rebalanced_f1_binary_classifier', 'rebalanced_f1_multi_classifier',
                       'not_rebalanced_f1_binary_classifier', 'not_rebalanced_f1_multi_classifier',
                       'rebalanced_accuracy_binary_classifier', 'rebalanced_accuracy_multi_classifier',
                       'not_rebalanced_accuracy_binary_classifier', 'not_rebalanced_accuracy_multi_classifier'
                       ]
            writer.writerow(headers)
            for experiment in comparison:
                writer.writerow([experiment[header] for header in headers])

    @staticmethod
    def save_models_as_h5(binary_dir: str, multi_dir: str, recurrent_dir: str):
        """
        Note:
        Assumes we have ran Assignments b, c, and d, with the last line commented out, such that they will save the best models

        Save three h5 files: one for each classifier. To save them, we load the best performing model as created by Evaluator.compare_design_choices.
        To select the best performing models, we select the best scores from Tables I-III in our paper.
        - Select models where Rebalance test = true, since the final test set will probably be balanced in such a way as well
        - Select model based on high test F1 score
        - File locations are based on the ordering of this table: 16 configurations in dirs 0-15 in a specific sub directory

        Chosen models:
        - Binary classifier:
            Best F1 score with RebalanceTrainVal=True, WeightedLoss=False
            Configuration index 5 in Table III
        - Multi classifier:
            Best F1 score with RebalanceTest=True gives config with ApplyBayes=True, RebalanceTest=True, RebalanceTrainVal=False, WeightedLoss=False
            Configuration index 3 in Table I
        - Recurrent classifier:
            Best F1 score with RebalanceTest=True gives config with ApplyBayes=True, RebalanceTest=True, RebalanceTrainVal=False, WeightedLoss=False
            Configuration index 3 in Table II
        @return:
        """

        binary_classifier_dir = f'{binary_dir}/5'
        multi_classifier_dir = f'{multi_dir}/3'
        recurrent_classifier_dir = f'{recurrent_dir}/3'

        binary_classifier = BinaryClassifier()
        multi_classifier = MultiClassifier()
        recurrent_classifier = RecurrentClassifier()

        binary_net = load_model(binary_classifier_dir, custom_objects={'f1': BaseClassifier.f1, 'loss': binary_classifier.loss()})
        multi_net = load_model(multi_classifier_dir, custom_objects={'f1': BaseClassifier.f1, 'loss': multi_classifier.loss()})
        recurrent_net = load_model(recurrent_classifier_dir, custom_objects={'f1': BaseClassifier.f1, 'loss': recurrent_classifier.loss()})

        binary_net.save('best_binary_classifier.h5')
        multi_net.save('best_multiclass_classifier.h5')
        recurrent_net.save('best_recurrent_classifier.h5')

        write_log('Saved best models')
