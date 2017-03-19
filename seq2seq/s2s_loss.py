import keras.backend as K


def s2sloss(mask):
    def loss(y_true, y_pred):
        return K.sum(- K.log(y_pred) * y_true, axis=-1) * K.equal(mask, 2)

    return loss
