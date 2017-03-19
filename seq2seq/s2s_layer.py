from keras.layers.recurrent import Recurrent
import keras.backend as K
from keras.engine import InputSpec
from keras import initializers
from keras import constraints, regularizers
from .backend import cumsum, zeros


class S2SLayer(Recurrent):
    """
    Layer for running something like an LSTM on an input sequence. Generates an output sequence.
    Predictions of output sequence are fed back to the LSTM.
    Predictions can be deterministic or stochastically sampled from a softmax.
    Input sequence and output sequences can be of different lengths within a batch.
    Input vector is 3d (batch, depth, 2).
      First component is input sequence concatenated to output sequence padded out to depth
      Second component is a mask, 1 if input sequence, 2 if output sequence, 0 if padding
    """

    def __init__(self, x_k, hidden_dim, stochastic=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.x_k = x_k
        self.hidden_dim = hidden_dim
        self.stochastic = stochastic

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.units = x_k + 1
        Recurrent.__init__(self, return_sequences=True, **kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(None, None, self.input_dim))
        self.state_spec = [InputSpec(shape=(None, self.hidden_dim)),
                           InputSpec(shape=(None, self.x_k + 1)),
                           InputSpec(shape=(None, 1))]
        self.depth = input_shape[1]
        self.states = [None, None, None]

        # Convenience functions for making parameters
        def W(shape, name):
            return self.add_weight(shape,
                                   name=name,
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)

        def b(shape, name):
            return self.add_weight(shape,
                                   name=name,
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        def pair(shape, name):
            return W(shape, "W{}".format(name)), b((shape[1],), "b{}".format(name))

        # h is hidden representation of inputs to the unit (hidden state and current character)
        Wh = W((self.hidden_dim, self.hidden_dim), 'Wh')
        Uh = W((self.x_k + 1, self.hidden_dim), 'Uh')
        bh = b((self.hidden_dim,), 'Ub')
        # f, i, o, c as standard LSTM
        Wf, bf = pair((self.hidden_dim, self.hidden_dim), "f")
        Wi, bi = pair((self.hidden_dim, self.hidden_dim), "i")
        Wo, bo = pair((self.hidden_dim, self.hidden_dim), "o")
        Wc, bc = pair((self.hidden_dim, self.hidden_dim), "c")
        # t and y are dense layers on top of LSTM to calculate the softmax
        Wt, bt = pair((self.hidden_dim, self.hidden_dim), "t")
        Wy, by = pair((self.hidden_dim, self.x_k + 1), "y")
        self.constants = (Wh, Uh, bh,
                          Wf, bf,
                          Wi, bi,
                          Wo, bo,
                          Wc, bc,
                          Wt, bt,
                          Wy, by
                          )
        self.built = True

    def get_constants(self, inputs, training=None):
        return self.constants

    def step(self, inputs, states):
        # Input is 3d
        # components are 0=input output sequences, 1=mask, 2=rng
        input_x = K.cast(inputs[:, 0], "int32")
        input_mask = K.cast(inputs[:, 1], "int32")
        rnd = inputs[:, 2]
        # states are (hidden state, softmax, output)
        hidden_t0 = states[0]
        softmax_t0 = states[1]
        output_t0 = states[2]
        # unpack parameters
        (Wh, Uh, bh,
         Wf, bf,
         Wi, bi,
         Wo, bo,
         Wc, bc,
         Wt, bt,
         Wy, by
         ) = states[3:]
        # switch depending on input_mask
        is_input = K.equal(input_mask, 1)
        is_training = K.equal(input_mask, 2)
        is_output = K.equal(input_mask, 3)
        # use input value if input or training, use previous output if generating
        x = (is_input + is_training) * input_x + is_output * output_t0
        # hidden representation combining current value and hidden state
        h = K.tanh(K.dot(hidden_t0, Wh) + Uh[x, :] + bh)
        # standard LSTM f,i,c,o
        f = K.sigmoid(K.dot(h, Wf) + bf)
        i = K.sigmoid(K.dot(h, Wi) + bi)
        c = K.tanh(K.dot(h, Wc) + bc)
        hidden_t1 = hidden_t0 * f + i * c
        # 2 layer dense network for softmax
        o = K.sigmoid(K.dot(h, Wo) + bo)
        tmp = K.tanh(K.dot(hidden_t1 * o, Wt) + bt)
        softmax_t1 = K.softmax(K.dot(tmp, Wy) + by)
        # use softmax to select an output
        if self.stochastic:
            # stochastic output
            csum = cumsum(softmax_t1, axis=1)
            output_t1 = rnd.dimshuffle((0, "x")) > csum
            output_t1 = K.sum(output_t1, axis=1)
        else:
            # use this instead to make deterministic
            output_t1 = K.argmax(softmax_t1, axis=1)
        output_t1 = K.cast(output_t1, "int32")
        output_mat = K.concatenate((softmax_t1, K.expand_dims(output_t1, axis=1)), axis=1)
        return output_mat, [hidden_t1, softmax_t1, output_t1]

    def get_initial_states(self, inputs):
        hidden0 = zeros((K.shape(inputs)[0], self.hidden_dim), dtype='float32')
        softmax0 = zeros((K.shape(inputs)[0], self.x_k + 1), dtype='float32')
        output0 = zeros((K.shape(inputs)[0],), dtype='int32')
        return [hidden0, softmax0, output0]
