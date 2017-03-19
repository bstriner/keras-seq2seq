import keras.backend as K


def s2sloss(mask):
    """
    Basically just categorical cross entropy masked for where the mask==2 (output)
    :param mask:
    :return:
    """

    def loss(y_true, y_pred):
        return K.sum(- K.log(y_pred) * y_true, axis=-1) * K.equal(mask, 2)

    return loss
