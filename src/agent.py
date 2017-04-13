"""
Testing ideas from RL^2 : Fast RL via Slow RL (OpenAI) on a multi armed bandit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import time


def generate_n_bandits(N):
    return np.random.rand(N)

def pull(bandits, n):
    return 1 if random.random() < bandits[n] else 0


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class RL2():
    def __init__(self, n_actions, statesize=None):
        self.n_actions = n_actions
        self.statesize = statesize

        # shape = batch_size, seq_length, action&reward&termination
        self.sequence_length = tf.placeholder(tf.int32)
        self.batch_size = tf.placeholder(tf.int32)
        if statesize:
            self.state_input = tf.placeholder(tf.float32, shape=[None, None, statesize])
        self.input = tf.placeholder(tf.float32, shape=[None, None, 3])
        self.embedded_input = self.embed_input()
        
        # defining the recurrent part
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(48)
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        
        self.rnn_output, self.rnn_output_state = tf.nn.dynamic_rnn(
                self.cell,
                self.embedded_input,
                initial_state=self.initial_state
        )

        # fully connected and output
        self.actions_distribution =  slim.fully_connected(self.rnn_output, self.n_actions, activation_fn=tf.nn.softmax, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
        self.value_function = tf.squeeze(slim.fully_connected(self.rnn_output, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None), 2)

        # easy endpoint for testing step by step (removes batch size and seq len)
        self.last_actions_distribution = tf.squeeze(self.actions_distribution)

        # loss function
        reward = self.reward = tf.placeholder(tf.float32, shape=[None, None]) # discounted sums

        self.value_loss = tf.reduce_sum(tf.square(reward-self.value_function))
        self.value_loss = tf.Print(self.value_loss, [self.value_loss], "Voss: ")
        

        self.entropy = - tf.reduce_sum(self.actions_distribution*tf.log(self.actions_distribution+1e-7))
        self.entropy_mul = tf.placeholder(tf.float32)
        #  values_plus = tf.concat(1, [self.values, tf.zeros((self.batch_size, 1))])
        #  self.advantage = reward + GAMMA * tf.slice(values_plus, [0, 1], [-1, -1]) - self.values
        #  self.advantage = discount(self.reward - self.value_function
        self.advantage = tf.placeholder(tf.float32, shape=[None, None])
        self.policy_loss = tf.reduce_sum(-tf.log(tf.reduce_sum(self.actions_distribution * self.chosen_actions, 2)
            +1e-7) * (self.advantage))
        self.loss = self.policy_loss + 0.5 * self.value_loss - self.entropy*0.05#self.entropy_mul # avoid log(0) with +1e-10
        self.loss = tf.Print(self.loss, [self.loss, self.loss/100], "Loss: ")
        #  total_loss = self.value_loss + self.loss
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def embed_input(self):
        self.action_choices = tf.cast(
            tf.slice(self.input, [0, 0, 0], [-1, -1, 1]),
            tf.int32)
        self.action_choices_OH = tf.squeeze(
                tf.one_hot(self.action_choices, self.n_actions),
                2)
        self.r_and_t = tf.slice(self.input, [0, 0, 1], [-1, -1, 2])
        self.chosen_actions = tf.slice(tf.concat(1, (self.action_choices_OH, tf.zeros((self.batch_size, 1, 2)))), [0, 1, 0], [-1, -1, -1])
        if self.statesize:
            return tf.cast(tf.concat(2, (self.action_choices_OH, self.r_and_t, self.state_input)), tf.float32)
        return tf.cast(tf.concat(2, (self.action_choices_OH, self.r_and_t)), tf.float32)


def discount(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    discounted_rewards[-1] = rewards[-1]
    for i in reversed(range(0, len(rewards)-1)):
        discounted_rewards[i] = rewards[i] + discounted_rewards[i+1] * gamma
    return discounted_rewards

