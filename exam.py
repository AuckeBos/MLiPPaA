from BinaryClassification import BinaryClassifier
from Evaluator import Evaluator
from MultiClassification import MultiClassifier


def assignment_a():
    classifier = BinaryClassifier()
    classifier.apply_bayes = True
    classifier.rebalance_distribution = [1 - 0.04, 0.04]
    net = Evaluator.evaluate(classifier)
    classifier.test(net)


def assignment_b():
    classifier = MultiClassifier()
    classifier.apply_bayes = True
    classifier.rebalance_distribution = [0.04] + ([(1 - 0.04) / 4] * 4)
    classifier.balance_training_set = False
    #
    # classifier.load_data()
    # net = classifier.compile_net()
    # classifier.test(net)

    net = Evaluator.evaluate(classifier)
    classifier.test(net)


assignment_a()
