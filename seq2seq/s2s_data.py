import numpy as np


def process_sequences(input_sequences, output_sequences, depth=None):
    """
    Expects lists of lists of ints
    Returns 3d int array (batch size, depth, [0=character, 1=mask])
    mask values: 1-input, 2-training output, 3-testing output, 0-padding
    :param sequences:
    :return:
    """
    assert (len(input_sequences) == len(output_sequences))
    if not depth:
        depth = max(len(i) + len(o) + 4 for i, o in zip(input_sequences, output_sequences))
    n = len(input_sequences)
    x = np.zeros((n, depth, 2), dtype=np.int32)
    for i, (iseq, oseq) in enumerate(zip(input_sequences, output_sequences)):
        idx = 0
        for c in iseq:
            x[i, idx, 0] = c + 1
            x[i, idx, 1] = 1
            idx += 1
        x[i, idx, 1] = 1
        idx += 1
        x[i, idx, 1] = 2
        idx += 1
        for c in oseq:
            x[i, idx, 0] = c + 1
            x[i, idx, 1] = 2
            idx += 1
    y = np.concatenate((x[:, 1:, 0], np.zeros((x.shape[0], 1), dtype=np.int32)), axis=1)
    x = np.concatenate((x, np.random.random((x.shape[0], x.shape[1], 1)).astype(np.float32)), axis=2)
    return x, y


def one_hot_2d(x, x_k):
    """
    Efficient one-hot encoding of 2d vector
    :param x:
    :param x_k:
    :return:
    """
    ret = np.zeros((x.shape[0], x.shape[1], x_k + 1), dtype=np.float32)
    r = np.repeat(np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=np.int32).reshape((-1, 1)), x.shape[1], axis=1)
    c = np.repeat(np.linspace(0, x.shape[1] - 1, x.shape[1], dtype=np.int32).reshape((1, -1)), x.shape[0], axis=0)
    ret[r.ravel(), c.ravel(), x.ravel()] = 1
    return ret


def process_test_sequences(input_sequences, depth):
    """
    Expects lists of lists of ints
    Returns 3d int array (batch size, depth, [0=character, 1=mask])
    :param sequences:
    :return:
    """
    n = len(input_sequences)
    x = np.zeros((n, depth, 2), dtype=np.int32)
    for i, iseq in enumerate(input_sequences):
        idx = 0
        for c in iseq:
            x[i, idx, 0] = c + 1
            x[i, idx, 1] = 1
            idx += 1
        x[i, idx, 1] = 1
        idx += 1
        x[i, idx, 1] = 2
        idx += 1
        while idx < depth:
            x[i, idx, 1] = 3
            idx += 1
    x = np.concatenate((x, np.random.random((n, depth, 1)).astype(np.float32)), axis=2)
    return x
