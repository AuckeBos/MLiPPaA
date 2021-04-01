import tensorflow as tf


class EvaluationCallback(tf.keras.callbacks.Callback):
    """
    This callback evaluates on the test set on each epoch end. The results are
    used for visualization purposes only (tensorboard)
    """

    def __init__(self, classifier):
        """
        The callback needs x and y of the test set
        :param classifier: The classifier
        """
        super().__init__()
        self.classifier = classifier
        self.log_dir = classifier.logdir

    def on_epoch_end(self, epoch, logs=None):
        """
        On epoch end, evaluate on the test set. Add results to tensorboard
        :param epoch:
        :param logs:
        """
        loss, f1, accuracy = self.classifier.test(self.model, False)
        tf.summary.create_file_writer(f'{self.log_dir}/test').set_as_default()
        logs['test_loss'] = loss
        logs['test_f1'] = f1
        logs['test_accuracy'] = accuracy
        tf.summary.scalar('epoch_loss', data=loss, step=epoch)
        tf.summary.scalar('epoch_f1', data=f1, step=epoch)
        tf.summary.scalar('epoch_accuracy', data=accuracy, step=epoch)
