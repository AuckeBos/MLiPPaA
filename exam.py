from BinaryClassification import BinaryClassifier
from Evaluator import Evaluator
from MultiClassification import MultiClassifier
from RecurrentClassification import RecurrentClassifier


def assignment_a():
    classifier = BinaryClassifier()
    # classifier.apply_bayes = True
    classifier.rebalance_test = [1 - 0.04, 0.04]
    classifier.rebalance_train_val = True
    net = Evaluator.evaluate(classifier)
    classifier.test(net)


def assignment_b():
    classifier = MultiClassifier()
    classifier.apply_bayes = False
    classifier.rebalance_test = None
    classifier.rebalance_train_val = None

    net = Evaluator.evaluate(classifier)
    classifier.test(net)

def assignment_d():
    classifier = RecurrentClassifier()
    classifier.apply_bayes = True
    classifier.rebalance_test = [0.04] + ([(1 - 0.04) / 4] * 4)
    net = Evaluator.evaluate(classifier)

# assignment_b()
assignment_d()
# Evaluator.compare_design_choices()
# Evaluator.evaluate_comparison('comparison.json')
