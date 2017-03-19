import os
from nltk.corpus import shakespeare
from keras.callbacks import LambdaCallback
from keras.layers import Input, Lambda
from keras.optimizers import Adam
import numpy as np
from keras.models import Model
from seq2seq.s2s_loss import s2sloss
from seq2seq.s2s_layer import S2SLayer
from seq2seq.s2s_data import process_sequences, process_test_sequences, one_hot_2d
import itertools


def clean_word(word):
    """
    Remove non-asci characters and downcase
    :param word:
    :return:
    """
    return "".join([c for c in word.lower() if ord(c) < 128])


def clean_words(words):
    """
    Remove words < 3 characters
    :param words:
    :return:
    """
    return [clean_word(w) for w in words if len(clean_word(w)) >= 3]


def shakespeare_words():
    """
    Concatenate all of shakespeare
    :return:
    """
    return itertools.chain.from_iterable(shakespeare.words(fileid) for fileid in shakespeare.fileids())


def get_charset(words):
    """
    List unique characters
    :param words:
    :return: list of characters, dictionary from characters to indexes
    """
    charset = list(set(itertools.chain.from_iterable(words)))
    charset.sort()
    charmap = {c: i for i, c in enumerate(charset)}
    return charset, charmap


def map_word(word, charmap):
    """
    Convert string to list of indexes into charset
    :param word:
    :param charmap:
    :return:
    """
    return [charmap[c] for c in word]


def map_words(words, charmap):
    return [map_word(w, charmap) for w in words]


def sample_bigram(vectors):
    """
    Sample random word pair
    :param vectors:
    :return:
    """
    i1 = np.random.randint(0, len(vectors) - 1)
    i2 = i1 + 1
    return vectors[i1], vectors[i2]


def sample_bigrams(vectors, n, x_k):
    """
    Sample n word pairs and convert to input/output vectors
    :param vectors:
    :param n:
    :param x_k:
    :return:
    """
    grams = [sample_bigram(vectors) for _ in range(n)]
    w1 = [w[0] for w in grams]
    w2 = [w[1] for w in grams]
    x, y = process_sequences(w1, w2)
    return x, one_hot_2d(y, x_k)


def bigram_generator(*args):
    """
    generator for fit_generator
    :param args:
    :return:
    """
    while True:
        yield sample_bigrams(*args)


def decode_vector(vector, charset):
    """
    List of indexes to a string
    :param vector:
    :param charset:
    :return:
    """
    return "".join(charset[x] for x in vector)


def decode_row(row, charset):
    """
    Output vector to a string
    :param row:
    :param charset:
    :return:
    """
    return "".join([charset[x - 1] if x > 0 else " " for x in row])


def decode_output(output, charset):
    """
    Output matrix to list of strings
    :param output:
    :param charset:
    :return:
    """
    return [decode_row(row, charset) for row in output]


def sample_word(vectors):
    i = np.random.randint(0, len(vectors))
    return vectors[i]


def sample_words(vectors, n):
    return [sample_word(vectors) for _ in range(n)]


def on_epoch_end(model, vectors, charset, depth):
    """
    Callback samples unigrams, generates word pair, and writes to file and stdout
    :param model:
    :param vectors:
    :param charset:
    :param depth:
    :return:
    """
    if not os.path.exists("output"):
        os.makedirs("output")

    def fun(epoch, logs):
        n = 32
        x = sample_words(vectors, n)
        xinp = process_test_sequences(x, depth)
        y = model.predict_on_batch(xinp).astype(np.int32)
        print "Epoch: {}".format(epoch)
        with open("output/epoch-{}.txt".format(epoch), 'w') as f:
            s = "Loss: {}".format(logs["loss"])
            print s
            f.write(s + "\n")
            for i in range(n):
                w1 = decode_vector(x[i], charset)
                w2 = (decode_row(y[i], charset)[len(w1)+1:]).split(" ")[0]
                s = "{}: [{}]".format(w1, w2)
                print s
                f.write(s + "\n")

    return fun


def main():
    # Hyperparameters
    hidden_dim = 512
    batch_size = 128
    steps_per_epoch = 512
    epochs = 1000
    lr = 1e-3

    # Load and clean data
    words = clean_words(shakespeare_words())
    charset, charmap = get_charset(words)
    x_k = len(charset)
    vectors = map_words(words, charmap)
    max_word = max(len(w) for w in vectors)
    depth = max_word * 2 + 4

    # Create model
    x = Input((None, 3), dtype='float32')
    s2s = S2SLayer(x_k, hidden_dim, stochastic=True)
    # output of layer is softmax and prediction concatenated; slice the output
    y = s2s(x)
    ysoftmax = Lambda(lambda z: z[:, :, :-1], output_shape=lambda z: (z[0], z[1], z[2] - 1))(y)
    ypred = Lambda(lambda z: z[:, :, -1], output_shape=lambda z: (z[0], z[1], 1))(y)
    # model for training
    m = Model(inputs=[x], outputs=[ysoftmax])
    m.summary()
    m.compile(Adam(lr), s2sloss(x[:, :, 1]))
    # model for testing
    mtest = Model(inputs=[x], outputs=[ypred])
    # callback to print results
    cb = LambdaCallback(on_epoch_end=on_epoch_end(mtest, vectors, charset, depth))
    # train model
    m.fit_generator(bigram_generator(vectors, batch_size, x_k), callbacks=[cb],
                    steps_per_epoch=steps_per_epoch, epochs=epochs,
                    verbose=1)


if __name__ == "__main__":
    main()
