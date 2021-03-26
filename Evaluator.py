from BaseClassification import BaseClassifier


class Evaluator:
    """
    Simple static class that provides evaluation functionality
    For now only provide the evaluate() method, which evaluates performance of a classifier
    """

    @staticmethod
    def evaluate(classifier: BaseClassifier):
        """
        Evaluate a classifier by training and testing it
        :param classifier:
        """
        net = classifier.train()
        classifier.test(net)
        return net
