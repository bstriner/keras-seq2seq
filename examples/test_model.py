import os
from nltk.corpus import shakespeare
from keras.callbacks import LambdaCallback
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from keras.models import Model
from seq2seq.s2s_loss import s2sloss
from seq2seq.s2s_layer import S2SLayer
from seq2seq.s2s_data import process_sequences, process_test_sequences
import itertools


def clean_word(word):
    return "".join([c for c in word.lower() if ord(c) < 128])


def clean_words(words):
    return [clean_word(w) for w in words if len(clean_word(w)) >= 3]


def shakespeare_words():
    return itertools.chain.from_iterable(shakespeare.words(fileid) for fileid in shakespeare.fileids())


def get_charset(words):
    charset = list(set(itertools.chain.from_iterable(words)))
    charset.sort()
    charmap = {c: i for i, c in enumerate(charset)}
    return charset, charmap


def map_word(word, charmap):
    return [charmap[c] for c in word]


def map_words(words, charmap):
    return [map_word(w, charmap) for w in words]


def sample_bigram(vectors):
    i1 = np.random.randint(0, len(vectors) - 1)
    i2 = i1 + 1
    return vectors[i1], vectors[i2]


def sample_bigrams(vectors, n, depth, x_k):
    grams = [sample_bigram(vectors) for _ in range(n)]
    w1 = [w[0] for w in grams]
    w2 = [w[1] for w in grams]
    x, y = process_sequences(w1, w2, depth)
    return x, targets(y, x_k)


def bigram_generator(*args):
    while True:
        yield sample_bigrams(*args)


def decode_vector(vector, charset):
    return "".join(charset[x] for x in vector)


def decode_row(row, charset):
    return "".join([charset[x - 1] if x > 0 else " " for x in row])


def decode_output(output, charset):
    return [decode_row(row, charset) for row in output]


def sample_word(vectors):
    i = np.random.randint(0, len(vectors))
    return vectors[i]


def sample_words(vectors, n):
    return [sample_word(vectors) for _ in range(n)]


def targets(x, x_k):
    ret = np.zeros((x.shape[0], x.shape[1], x_k + 1), dtype=np.float32)
    r = np.repeat(np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=np.int32).reshape((-1, 1)), x.shape[1], axis=1)
    c = np.repeat(np.linspace(0, x.shape[1] - 1, x.shape[1], dtype=np.int32).reshape((1, -1)), x.shape[0], axis=0)
    ret[r.ravel(), c.ravel(), x.ravel()] = 1
    return ret


def on_epoch_end(model, vectors, charset, depth):
    if not os.path.exists("output"):
        os.makedirs("output")

    def fun(epoch, logs):
        n = 32
        x = sample_words(vectors, n)
        xinp = process_test_sequences(x, depth)
        y = model.predict_on_batch(xinp)
        y = np.argmax(y, axis=-1)
        print "Epoch: {}".format(epoch)
        with open("output/epoch-{}.txt".format(epoch), 'w') as f:
            for i in range(n):
                w1 = decode_vector(x[i], charset)
                w2 = decode_row(y[i], charset) #[len(w1) + 1:].strip()
                s = "{}: [{}]".format(w1, w2)
                print s
                f.write(s + "\n")

    return fun


def main():
    words = clean_words(shakespeare_words())
    charset, charmap = get_charset(words)
    vectors = map_words(words, charmap)
    max_word = max(len(w) for w in vectors)
    depth = max_word * 2 + 4
    print ("Depth: {}, Charset: {}".format(depth, len(charset)))
    x_k = len(charset)
    x = Input((depth, 2), dtype='int32')
    hidden_dim = 512
    s2s = S2SLayer(x_k, hidden_dim)
    y = s2s(x)
    m = Model(inputs=[x], outputs=[y])
    m.summary()
    m.compile(Adam(1e-3), s2sloss(x[:, :, 1]))
    batch_size = 64
    cb = LambdaCallback(on_epoch_end=on_epoch_end(m, vectors, charset, depth))
    m.fit_generator(bigram_generator(vectors, batch_size, depth, x_k), callbacks=[cb], steps_per_epoch=256, epochs=100)


if __name__ == "__main__":
    main()
