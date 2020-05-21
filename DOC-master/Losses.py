

from keras import backend as k
import tensorflow as tf


lambda_coefficient = 0.1


def compactnes_loss(target, pred):
    n_dim = tf.shape(pred)[0] # number of features vecs
    k_dim = tf.shape(pred)[1] # feature vec dim
    mean = k.mean(pred, axis=0)
    diff_step = pred - mean
    sample_variance = k.sum(k.square(diff_step), axis=1)
    var_sum = k.sum(sample_variance)

    return tf.cast((n_dim / (k_dim * k.pow(n_dim - 1, 2))),
                               tf.float32) * var_sum


descriptiveness_loss = lambda target, pred: tf.keras.losses.categorical_crossentropy(target, pred)