from BinaryClassification import BinaryClassifier
from Evaluator import Evaluator
from MultiClassification import MultiClassifier
from RecurrentClassification import RecurrentClassifier


def assignment_a():
    """
    Assignment a) consists of our simple baseline BinaryClassifier
    In this function we train and evaluate it.
    Note we can generate an evaluation csv for different configurations via our Evaluator (comment out the last line)
    """
    classifier = BinaryClassifier()
    classifier.apply_bayes = True
    classifier.rebalance_test = [1 - 0.04, 0.04]
    classifier.rebalance_train_val = None
    Evaluator.evaluate(classifier)
    # Evaluator.compare_design_choices('binary', './Evaluations/BinaryClassifierEvaluation')


def assignment_b():
    """
    Assignment b) consists of our MultiClassifier
    In this function we train and evaluate it.
    Note we can generate an evaluation csv for different configurations via our Evaluator (comment out the last line)
    """
    classifier = MultiClassifier()
    classifier.apply_bayes = True
    classifier.rebalance_test = [0.04] + ([(1 - 0.04) / 4] * 4)
    classifier.rebalance_train_val = None
    Evaluator.evaluate(classifier)
    # Evaluator.compare_design_choices('multi', './Evaluations/MultiClassifierEvaluation')


def assignment_c():
    """
    Assignment c) compares our MultiClassifier with our BinaryClassifier

    Compare both classifiers by fully training them on all configurations. Then generate a json with a binary comparison, and convert it to a csv with the
    relevant testing data
    """
    Evaluator.compare_design_choices('multi', './Evaluations/MultiClassifierEvaluation')
    Evaluator.compare_design_choices('binary', './Evaluations/BinaryClassifierEvaluation')
    Evaluator.compare_binary_performance('./Evaluations/BinaryClassifierEvaluation', './Evaluations/MultiClassifierEvaluation')
    Evaluator.binary_comparison_to_csv('./binary_comparison.json')


def assignment_d():
    """
    Assignment d) creates a more sophisticated classifier. We chose to build a recurrent classifier using LSTMs.
    In this function we train and evaluate it.
    Note we can generate an evaluation csv for different configurations via our Evaluator (comment out the last line)
    @return:
    """
    classifier = RecurrentClassifier()
    classifier.apply_bayes = True
    classifier.rebalance_test = [0.04] + ([(1 - 0.04) / 4] * 4)
    classifier.rebalance_train_val = None
    Evaluator.evaluate(classifier)
    # Evaluator.compare_design_choices('recurrent', './Evaluations/RecurrentClassifierEvaluation')

# Uncomment to run one of the above functions
# assignment_a()
# assignment_b()
# assignment_c()
# assignment_d()
