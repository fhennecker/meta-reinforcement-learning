"""
Testing ideas from RL^2 : Fast RL via Slow RL (OpenAI) on a multi armed bandit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random


def generate_n_bandits(N):
    return np.random.rand(N)

def pull(bandits, n):
    return 1 if random.random() < bandits[n] else 0


class RL2():
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits

        # shape = batch_size, seq_length, action&reward&termination
        self.input = tf.placeholder(tf.float32, shape=[1, 1, 3])
        
        # defining the recurrent part
        self.cell = tf.nn.rnn_cell.GRUCell(10)
        self.initial_state = self.cell.zero_state(1, tf.float32)
        
        self.rnn_output, self.rnn_output_state = tf.nn.dynamic_rnn(
                self.cell,
                self.input,
                initial_state=self.initial_state
        )

        # fully connected and output
        self.rnn_output = tf.squeeze(self.rnn_output, axis=[0])
        self.actions_distribution = tf.squeeze(tf.nn.softmax(
                slim.fully_connected(self.rnn_output, self.n_bandits)
        ))

        # loss function
        self.action = tf.placeholder(tf.int32)
        action_one_hot = tf.one_hot(self.action, self.n_bandits)
        self.reward = tf.placeholder(tf.float32)
        self.loss = -tf.log(tf.reduce_sum
            (self.actions_distribution * self.reward * action_one_hot)+1e-10)
        self.train_step = tf.train.RMSPropOptimizer(1e-2).minimize(self.loss)




def train():
    n_bandits = 5
    n_trials = int(1e1)
    n_episodes_per_trial = 10
    episode_length = 100

    nn = RL2(n_bandits)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for t in range(n_trials):

            # generate a new MDP
            bandits = generate_n_bandits(n_bandits)
            print("New trial with bandits", bandits)
            print("(Supposed to choose %d)" % np.argmax(bandits))
            # the hidden state is passed from episode to episode
            hidden_state = None

            for e in range(n_episodes_per_trial):

                initial_action = random.randint(0, n_bandits-1)
                initial_reward = pull(bandits, initial_action)
                rnn_input = np.array([initial_action, initial_reward, 0])

                for i in range(episode_length):
                    feed_dict = {
                            nn.input : [[rnn_input]],
                            nn.action : rnn_input[0],
                            nn.reward : rnn_input[1],
                    }
                    if hidden_state is not None:
                        feed_dict[nn.initial_state] = hidden_state

                    actions_distribution, hidden_state, _, loss = sess.run(
                            [nn.actions_distribution, nn.rnn_output_state, 
                                nn.train_step, nn.loss], 
                            feed_dict)

                    action = np.random.choice(
                            range(n_bandits), p=actions_distribution)
                    reward = pull(bandits, action)
                    terminated = 1 if i == episode_length-1 else 0
                    rnn_input = np.array([action, reward, terminated])
                    if i == episode_length -1:
                        print(loss, np.argmax(actions_distribution))

if __name__ == "__main__":
    train()


