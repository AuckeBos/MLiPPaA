from BinaryClassification import BinaryClassifier
from Evaluator import Evaluator
from MultiClassification import MultiClassifier


def assignment_a():
    classifier = BinaryClassifier()
    Evaluator.evaluate(classifier)


def assignment_b():
    classifier = MultiClassifier()
    classifier.apply_bayes = True
    classifier.balance_test_set = [0.04] + ([(1 - 0.04) / 4] * 4)
    classifier.balance_validation_set = classifier.balance_test_set
    Evaluator.evaluate(classifier)


assignment_b()
