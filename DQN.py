import tensorflow as tf
import numpy as np
import collections


class DQN():

    def __init__(self, state_dim, num_actions, use_target_network=True, use_double_dqn=True, gamma=0.99, hidden_size=64,
                 batch_size=128, max_length=32):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.memory = ReplayBuffer()
        self.use_target_network = use_target_network
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_seq_length = max_length

        self.qvars = None
        self.tvars = None

        self.input = tf.placeholder(tf.float32, shape=[None, self.max_seq_length, self.state_dim])
        self.input_state_length = tf.placeholder(tf.int32, shape=[None, ])

        # q network graph definition
        with tf.variable_scope("qnet"):
            self.output = self._network(self.input)
            current_scope = tf.get_default_graph().get_name_scope()
            self.qvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
            self.qvars.sort(key=lambda x: x.name)

        if self.use_target_network or self.use_double_dqn:
            # target network graph definition
            with tf.variable_scope("tnet"):
                self.target = self._network(self.input)
                current_scope = tf.get_default_graph().get_name_scope()
                self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
                self.tvars.sort(key=lambda x: x.name)

            # target network weights update operations
            self.update_target_op = [var[0].assign(var[1]) for var in zip(self.tvars, self.qvars)]

        # training operations definition
        self.yt_loss = tf.placeholder(tf.float32, shape=[None, ])
        self.actions = tf.placeholder(tf.int32, shape=[None, ])
        actions_onehot = tf.one_hot(self.actions, num_actions)
        q_actions = tf.multiply(actions_onehot, self.output)

        self.loss = tf.losses.huber_loss(self.yt_loss, tf.reduce_sum(q_actions, axis=1))
        self.train_op = tf.train.AdamOptimizer.minimize(loss=self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _network(self, input_state):
        basic_cell = tf.contrib.GRUCell(num_units=self.hidden_size)
        _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32,
                                      sequence_length=self.input_state_length)
        net = tf.contrib.slim.fully_connected(states, self.hidden_size)
        net = tf.contrib.slim.fully_connected(net, self.hidden_size)
        output = tf.contrib.slim.fully_connected(net, self.num_actions, activation_fn=None)
        return output

    # update target network weights
    def updateTargetNetwork(self):
        if self.use_target_network or self.use_double_dqn:
            self.sess.run(self.update_target_op)

    def updateModel(self):
        states, state_length, actions, rewards, new_states, new_states_length, endgames = self.memory.sample(self.batch_size)

        qtarget = None

        if self.use_double_dqn:
            # computing target Q value using target network and double deep q-network algorithm
            [n_out, t_out] = self.sess.run([self.output, self.target],
                                           feed_dict={self.input: new_states,
                                                      self.input_state_length: new_states_length})
            target_action = np.argmax(n_out, axis=1)
            qtarget = np.array([output_sample[target_action[sample]] for sample, output_sample in enumerate(t_out)])
        elif self.use_target_network:
            # computing target Q value using target network
            qtarget = np.amax(self.sess.run(self.target, feed_dict={self.input: new_states,
                                                                    self.input_state_length: new_states_length}), axis=1)
        else:
            qtarget = np.amax(self.sess.run(self.output, feed_dict={self.input: new_states,
                                                                    self.input_state_length: new_states_length}), axis=1)

        yt = rewards + self.gamma * (np.logical_not(endgames) * qtarget)

        # computing loss and update weights of  Q network
        loss, _ = self.sess.run([self.loss, self.train_op],
                                feed_dict={self.input: states, self.input_state_length: state_length, self.yt_loss: yt,
                                           self.actions: np.array(actions)})
        return loss

    def get_action(self, state, state_length):
        state = np.reshape(state, [state_length, self.state_dim])
        return self.sess.run(self.output, feed_dict={self.input: state, self.input_state_length:state_length})

    def store(self, s1, s1_length, a, r, s2, s2_length, done):
        s1 = np.array(s1).reshape([s1_length, self.state_dim])
        s2 = np.array(s2).reshape([s2_length, self.state_dim])
        if s1_length < self.max_seq_length:  # 补0
            padding_mat = np.zeros([self.max_seq_length - s1_length, self.state_dim])
            s1 = np.vstack((s1, padding_mat))
        if s2_length < self.max_seq_length:
            padding_mat = np.zeros([self.max_seq_length - s2_length, self.state_dim])
            s2 = np.vstack((s2, padding_mat))
        self.replay_buffer.add((s1, s1_length, a, r, s2, s2_length, done))


class DuelingDQN(DQN):

    # just redefine the network architecture
    def _network(self, input_state):
        basic_cell = tf.contrib.GRUCell(num_units=self.hidden_size)
        _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32,
                                      sequence_length=self.input_state_length)
        net = tf.contrib.slim.fully_connected(states, self.hidden_size)
        net = tf.contrib.slim.fully_connected(net, self.hidden_size)
        fc_a = tf.contrib.slim.fully_connected(net, self.hidden_size)
        fc_v = tf.contrib.slim.fully_connected(net, self.hidden_size)
        advantage = tf.contrib.slim.fully_connected(fc_a, self.num_actions)
        value = tf.contrib.slim.fully_connected(fc_v, 1, activation_fn=None)
        output = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return output


class ReplayBuffer(object):
    def __init__(self, max_len=100000):
        self.storage = collections.deque(maxlen=max_len)

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=32):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s1, s1_length, a, r, s2, s2_length, done = [], [], [], [], [], [], []

        for i in ind:
            d1, d2, d3, d4, d5, d6, d7 = self.storage[i]
            s1.append(np.array(d1, copy=False))
            s1_length.append(np.array(d2, copy=False))
            a.append(np.array(d3, copy=False))
            r.append(np.array(d4, copy=False))
            s2.append(np.array(d5, copy=False))
            s2_length.append(np.array(d6, copy=False))
            done.append(np.array(d7, copy=False))

        return np.array(s1), np.array(s1_length), np.array(a), np.array(r), np.array(s2), np.array(s2_length), np.array(
            done)

    def get_size(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
