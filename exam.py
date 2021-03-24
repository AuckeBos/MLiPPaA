from BinaryClassification import BinaryClassifier
from Evaluator import Evaluator
from MultiClassification import MultiClassifier


def assignment_a():
    classifier = BinaryClassifier()
    Evaluator.evaluate(classifier)


def assignment_b():
    classifier = MultiClassifier()
    Evaluator.evaluate(classifier)


# assignment_b()
