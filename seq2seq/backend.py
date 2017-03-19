import keras.backend as K

if K.backend() == "theano":
    import theano.tensor as T


    def cumsum(x, axis=-1):
        return T.cumsum(x, axis=axis)


    def zeros(shape, dtype='float32'):
        """
        Not the same as K.zeros. Creates dynamic tensor so can work on dynamic shapes.
        :param shape:
        :param dtype:
        :return:
        """
        return T.zeros(shape, dtype=dtype)
else:
    import tensorflow as tf


    def cumsum(x, axis=-1):
        return tf.cumsum(x, axis=axis)


    def zeros(shape, dtype='float32'):
        return tf.zeros(shape, dtype=dtype)
