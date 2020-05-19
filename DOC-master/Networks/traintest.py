import tensorflow as tf


class Trainer:
    """
    Manage the train step
    """
    def __init__(self, model, optimizer, loss_lambda, D_loss, C_loss):
        self.model = model
        self.optimizer = optimizer
        self.end_training = False
        self.train_D_loss = tf.keras.metrics.Mean(name='D_loss')
        self.train_C_loss = tf.keras.metrics.Mean(name='C_loss')
        self.D_loss = D_loss
        self.C_loss = C_loss
        self.loss_lambda = loss_lambda

    def get_step(self):

        @tf.function()
        def train_step(ref_data, ref_labels, tar_data):
            with tf.GradientTape() as tape:

                # Descriptiveness loss
                prediction = self.model(ref_data)
                l_D = self.total_loss(0, prediction, ref_labels)
                self.train_D_loss(l_D)

                # Compactness loss
                prediction = self.model(tar_data)
                l_C = self.total_loss(1, prediction, ref_labels)
                self.train_C_loss(l_C)

            D_gradients = tape.gradient(l_D, self.model.trainable_variables)
            C_gradients = tape.gradient(l_C, self.model.trainable_variables)
            total_gradient = (1 - self.loss_lambda) * D_gradients + self.loss_lambda * C_gradients

            self.optimizer.apply_gradients([(total_gradient, self.model.trainable_variables)])

        return train_step

    def total_loss(self, alpha, pred, labels):
        return (1-alpha) * self.D_loss(labels, pred) + alpha * self.C_loss(labels, pred)

