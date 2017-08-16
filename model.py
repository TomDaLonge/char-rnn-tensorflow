import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # select the type of the rnn
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # create the rnn (layers/cells) itself
        with tf.name_scope("CreateCells"):
            cells = []
            for _ in range(args.num_layers):
                cell = cell_fn(args.rnn_size)
                if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                    cell = rnn.DropoutWrapper(cell,
                                              input_keep_prob=args.input_keep_prob,
                                              output_keep_prob=args.output_keep_prob)
                cells.append(cell)

            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # create placeholders for input/target data, coming from the other parts of the program
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length], "InputData")
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length], "Targets")

        # intitial state of the rnn
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # this is the last layer
        with tf.variable_scope("LastLayer"):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        # map the signs/letters from the input to number tensors in a lookup table
        with tf.name_scope("CreateEmbedding"):
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        with tf.name_scope("Dropout"):
            # dropout beta testing: double check which one should affect next line
            if training and args.output_keep_prob:
                inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        # change dimensions of the input data tensor of the rnn
        with tf.name_scope("ResizeInput"):
            inputs = tf.split(inputs, args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # to be used in the next step
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # propagate the data through the rnn
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='LSTMCells')

        # change dimensions of the output data tensor of the rnn
        with tf.name_scope("ReshapeOutput"):
            output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # calculations in the last layer
        with tf.name_scope("CreateLogits"):
            self.logits = tf.matmul(output, softmax_w) + softmax_b
        # how likely is it for a sign/letter to appear next? used for sampling later
        with tf.name_scope("CreateProbabilities"):
            self.probs = tf.nn.softmax(self.logits)

        # calculate the losses for this sequence
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        # mean loss
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        # create gradients with backpropagation, clip them and optimize the weights
        with tf.name_scope("CreateGradients"):
            rawgrads = tf.gradients(self.cost, tvars)
        with tf.name_scope("GradientClipping"):
            grads, _ = tf.clip_by_global_norm(rawgrads, args.grad_clip)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument for tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
