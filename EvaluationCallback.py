import tensorflow as tf


class EvaluationCallback(tf.keras.callbacks.Callback):
    """
    This callback evaluates on the test set on each epoch end. The results are
    used for vizualisation purposes only (tensorboard)
    """

    def __init__(self, x, y, log_dir):
        """
        The callback needs x and y of the test set
        :param x:
        :param y:
        :param log_dir:
        """
        super().__init__()
        self.x = x
        self.y = y
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        """
        On epoch end, evaluate on the test set. Add loss and accuracy to tensorboard
        :param epoch:
        :param logs:
        """
        results = self.model.evaluate(self.x, self.y, verbose=0)
        tf.summary.create_file_writer(f'{self.log_dir}/test').set_as_default()
        tf.summary.scalar('epoch_loss', data=results[0], step=epoch)
        tf.summary.scalar('epoch_accuracy', data=results[1], step=epoch)
