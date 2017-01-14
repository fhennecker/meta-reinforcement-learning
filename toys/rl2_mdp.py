"""
Testing ideas from RL^2 : Fast RL via Slow RL (OpenAI) on a multi armed bandit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt


def generate_n_bandits(N):
    return np.random.rand(N)

def pull(bandits, n):
    return 1 if random.random() < bandits[n] else 0


class RL2():
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits

        # shape = batch_size, seq_length, action&reward&termination
        self.sequence_length = tf.placeholder(tf.int32)
        self.input = tf.placeholder(tf.float32, shape=[1, None, 2])
        self.embedded_input = self.embed_input()
        
        # defining the recurrent part
        self.cell = tf.nn.rnn_cell.GRUCell(200)
        self.initial_state = self.cell.zero_state(1, tf.float32)
        
        self.rnn_output, self.rnn_output_state = tf.nn.dynamic_rnn(
                self.cell,
                self.input,
                initial_state=self.initial_state
        )

        # fully connected and output
        self.actions_distribution = tf.nn.softmax(
                slim.fully_connected(self.rnn_output, self.n_bandits)
        )

        # easy endpoint for testing step by step (removes batch size and seq len)
        self.last_actions_distribution = tf.squeeze(self.actions_distribution)

        # loss function
        self.reward = tf.placeholder(tf.float32)
        self.loss = -tf.log(tf.reduce_sum(
                self.actions_distribution * self.reward * self.action_choices_OH
            ) +1e-10) # avoid log(0) with +1e-10
        self.train_step = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss)

    def embed_input(self):
        self.action_choices = tf.cast(
            tf.slice(self.input, [0, 0, 0], [-1, -1, 1]),
            tf.int32)
        self.action_choices_OH = tf.squeeze(
                tf.one_hot(self.action_choices, self.n_bandits),
                2)
        self.termination_flags = tf.slice(self.input, [0, 0, 1], [-1, -1, 1])
        return tf.concat(2, (self.action_choices_OH, self.termination_flags))




def train():
    distribution_size = 10
    n_bandits = 5
    n_trials = int(3e3)
    n_episodes_per_trial = 10
    episode_length = 100

    nn = RL2(n_bandits)
    image = np.zeros((300, episode_length))
    pull_image = np.zeros((300, episode_length))
    image_line = 0

    bandits_distrib = [generate_n_bandits(n_bandits) for i in range(distribution_size)]

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        plt.ion()
        total_trial_rewards = []
        for t in range(n_trials):

            # generate a new MDP
            bandits = random.choice(bandits_distrib)
            print("New trial with bandits", bandits)
            print("(Supposed to choose %d with value %f)" % (np.argmax(bandits), np.max(bandits)))
            # the hidden state is passed from episode to episode
            hidden_state = None
            start_hidden_state = hidden_state
            total_trial_reward = 0


            for e in range(n_episodes_per_trial):

                total_episode_reward = 0
                episode_distributions = []

                initial_action = random.randint(0, n_bandits-1)
                rnn_input = np.array([initial_action, 0])

                # inputs will be used to update the network based on a full
                # episode
                inputs = [rnn_input]

                for i in range(episode_length):

                    # get action probabilities from network
                    feed_dict = { 
                            nn.sequence_length: 1,
                            nn.input : [[rnn_input]] 
                    }
                    if hidden_state is not None:
                        feed_dict[nn.initial_state] = hidden_state
                    if i == 0 :
                        start_hidden_state = hidden_state

                    actions_distribution, hidden_state = sess.run(
                            [nn.last_actions_distribution, nn.rnn_output_state], 
                            feed_dict)
                    episode_distributions.append(actions_distribution)

                    # choosing action
                    action = np.random.choice(
                            range(n_bandits), p=actions_distribution)

                    # record reward
                    reward = pull(bandits, action)
                    total_episode_reward += reward

                    # build input for next step
                    terminated = 1 if i == episode_length-1 else 0
                    rnn_input = np.array([action, terminated])
                    inputs.append(rnn_input)
                    
                if e%5 == 0 and image_line < image.shape[0]:
                    pulls = np.array(inputs)[1:,0].reshape([1, -1])
                    optimal_pulls = pulls == np.argmax(bandits)
                    quality = np.argsort(bandits)[pulls]
                    image[image_line, :] = quality
                    pull_image[image_line, :] = pulls
                    image_line += 1
                    plt.figure(1)
                    plt.imshow(image, cmap='gray', interpolation='nearest')
                    plt.draw()
                    plt.pause(0.001)
                    plt.figure(3)
                    plt.imshow(pull_image, cmap='gray', interpolation='nearest')
                    plt.draw()
                    plt.pause(0.001)

                feed_dict={
                    nn.sequence_length : episode_length,
                    nn.input : np.array([inputs]),
                    nn.reward : total_episode_reward
                }
                if start_hidden_state is not None:
                    feed_dict[nn.initial_state] = start_hidden_state
                loss, _ = sess.run([nn.loss, nn.train_step], feed_dict)
                print(loss, total_episode_reward)

                total_trial_reward += total_episode_reward

            total_trial_reward /= n_episodes_per_trial
            total_trial_rewards.append(total_trial_reward)
            print(total_trial_reward)
            plt.figure(2)
            plt.clf()
            plt.plot(total_trial_rewards)
            plt.pause(0.001)
            plt.show()

if __name__ == "__main__":
    train()


