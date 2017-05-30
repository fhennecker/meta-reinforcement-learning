"""
Testing ideas from RL^2 : Fast RL via Slow RL (OpenAI) on a multi armed bandit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import time


GAMMA = 0.8

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
    def __init__(self, n_arms):
        self.n_arms = n_arms

        # shape = batch_size, seq_length, action&reward&termination
        self.sequence_length = tf.placeholder(tf.int32)
        self.batch_size = tf.placeholder(tf.int32)
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
        self.actions_distribution =  slim.fully_connected(self.rnn_output, self.n_arms, activation_fn=tf.nn.softmax, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
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
                tf.one_hot(self.action_choices, self.n_arms),
                2)
        self.r_and_t = tf.slice(self.input, [0, 0, 1], [-1, -1, 2])
        self.chosen_actions = tf.slice(tf.concat(1, (self.action_choices_OH, tf.zeros((self.batch_size, 1, 2)))), [0, 1, 0], [-1, -1, -1])
        return tf.cast(tf.concat(2, (self.action_choices_OH, self.r_and_t)), tf.float32)


def discount(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    discounted_rewards[-1] = rewards[-1]
    for i in reversed(range(0, len(rewards)-1)):
        discounted_rewards[i] = rewards[i] + discounted_rewards[i+1] * gamma
    return discounted_rewards



def test():
    with tf.Session() as sess:
        nn = RL2(2)
        saver = tf.train.Saver()
        saver.restore(sess, './training/agent-7000')
        total_reward_per_bandit = []

        diff_values = 20
        n_eps_per_value = 100
        for b in range(diff_values):
            total_reward_per_bandit.append(0)
            bandit = [0.5+b/(diff_values*2), 0.5-b/(diff_values*2)]
            print(bandit)

            for k in range(n_eps_per_value):
                random.shuffle(bandit)
                choice = random.randint(0,1)
                reward = pull(bandit, choice)
                hidden_state = None
                optimal = []
                
                total_episode_reward = 0
                for i in range(100):
                    feed_dict = { 
                            nn.batch_size : 1,
                            nn.sequence_length: 1,
                            nn.input : [[[choice, i, reward]] ]
                    }
                    if hidden_state is not None:
                        feed_dict[nn.initial_state] = hidden_state
                    if i == 0 :
                        start_hidden_state = hidden_state

                    actions_distribution, hidden_state = sess.run(
                            [nn.last_actions_distribution, nn.rnn_output_state], feed_dict)

                    choice = np.random.choice(2, p=actions_distribution)
                    reward = pull(bandit, choice)
                    total_episode_reward += reward
                    optimal.append(choice == np.argmax(bandit))
                total_reward_per_bandit[-1] += total_episode_reward
        total_reward_per_bandit = np.array(total_reward_per_bandit)/n_eps_per_value

        p_bs = [0.5+b/(diff_values*2) for b in range(diff_values)]
        plt.plot(p_bs, [(0.5+b/(diff_values*2))*100 for b in range(diff_values)], 'm--')
        plt.plot(p_bs, total_reward_per_bandit, color=(0.28, 0.6, 0.85))
        plt.legend(['Best expected reward', 'Actual reward'], loc='upper left')
        plt.ylabel('Average trial reward')
        plt.xlabel('$p_b$')
        plt.xlim([0.5, 1])
        plt.savefig('bandit_test.pdf')
        plt.show()









def train():
    tf.reset_default_graph()
    distribution_size = 10
    n_arms = 2
    n_trials = int(100e3)
    n_episodes_per_trial = 1
    episode_length = 100

    nn = RL2(n_arms)
    image = np.zeros((400, episode_length))
    pull_image = np.zeros((400, episode_length))
    image_line = 0

    start_entropy = 1.0
    end_entropy = 0.0
    end_entropy_iteration = 1

    summary_writer = tf.summary.FileWriter('bonsoir4')

    bandits_distrib = [generate_n_bandits(n_arms) for i in range(distribution_size)]

    def testb(i):
        res = [j/10 for j in range(n_arms)]
        random.shuffle(res)
        res[i] = 0.9
        return res

    bandits_distrib = [testb(i) for i in range(n_arms)]

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        tw = tf.summary.FileWriter('/tmp/thesistrain2', sess.graph)
        plt.ion()
        total_trial_rewards = []
        for t in range(n_trials):

            # generate a new MDP
            #  bandits = random.choice(bandits_distrib)
            # the hidden state is passed from episode to episode
            hidden_state = None
            start_hidden_state = hidden_state
            total_trial_reward = 0


            inputs = [[] for i in range(n_episodes_per_trial)]
            rewards = [[] for i in range(n_episodes_per_trial)]
            values = [[] for i in range(n_episodes_per_trial)]

            for e in range(n_episodes_per_trial):

                total_episode_reward = 0
                episode_distributions = []

                hidden_state = None
                #  bandits = generate_n_bandits(n_arms)
                bandits = random.choice([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
                #  random.shuffle(bandits)
                print(bandits)

                action = random.randint(0, n_arms-1)
                reward = pull(bandits, action)
                rnn_input = np.array([action, 0, reward])

                for i in range(episode_length):
                    inputs[e].append(rnn_input)
                    rewards[e].append(reward)

                    # get action probabilities from network
                    feed_dict = { 
                            nn.batch_size : 1,
                            nn.sequence_length: 1,
                            nn.input : [[rnn_input]] 
                    }
                    if hidden_state is not None:
                        feed_dict[nn.initial_state] = hidden_state
                    if i == 0 :
                        start_hidden_state = hidden_state

                    actions_distribution, hidden_state, value = sess.run(
                            [nn.last_actions_distribution, nn.rnn_output_state, nn.value_function], 
                            feed_dict)
                    episode_distributions.append(actions_distribution)
                    values[e].append(value[0][0])

                    # choosing action
                    action = np.random.choice(
                            range(n_arms), p=actions_distribution)

                    reward = pull(bandits, action)
                    total_episode_reward += reward

                    # build input for next step
                    terminated = 1 if i == episode_length-1 else 0
                    rnn_input = np.array([action, i, reward])
                    
                total_trial_reward += total_episode_reward

            print(np.array(inputs).shape)
            if t%20 == 0 and image_line < image.shape[0]:
                pulls = np.array(inputs[-1])[:,0].reshape([1, -1])
                optimal_pulls = pulls == np.argmax(bandits)
                quality = np.argsort(bandits)[pulls]
                image[image_line, :] = quality
                pull_image[image_line, :] = pulls
                image_line += 1
                start = time.time()
            if t%100 == 0:
                plt.figure(1)
                plt.imshow(image, cmap='gray', interpolation='nearest')
                plt.ylabel('Training trial (x20 trials)')
                plt.xlabel('Episode number within the trial')
                plt.draw()
                plt.pause(0.001)
                plt.figure(3)
                plt.ylabel('Training trial (x20 trials)')
                plt.xlabel('Episode number within the trial')
                plt.imshow(pull_image, cmap='gray', interpolation='nearest')
                plt.draw()
                plt.pause(0.001)
                image.tofile(open('figreport/image.txt', 'w'))
                pull_image.tofile(open('figreport/pull_image.txt', 'w'))
            print(time.time()-start)

            entropymul = max(end_entropy, start_entropy - (start_entropy-end_entropy) * t/ end_entropy_iteration)
            print("ENTROPY :", entropymul)
            learning_rate = 1e-3
            #  if t > 500:
                #  learning_rate = 1e-4
            #  if t > 1000:
                #  learning_rate = 1e-5
            value_plus = values
            for v in value_plus:
                v.append(0)
            advantages = []
            for e in range(n_episodes_per_trial):
                e_advantages = np.array(rewards[e]) + GAMMA * np.array(value_plus[e][1:]) - np.array(value_plus[e][:-1])
                e_advanatages = discount(e_advantages, GAMMA)
                advantages.append(e_advantages)

            feed_dict={
                nn.batch_size : n_episodes_per_trial,
                nn.sequence_length : episode_length,
                nn.input : np.array(inputs),
                nn.reward : np.array(list(map(lambda r:discount(r, GAMMA), rewards))),
                nn.entropy_mul : entropymul,
                nn.learning_rate : learning_rate,
                nn.advantage : advantages
            }
            #  if start_hidden_state is not None:
                #  feed_dict[nn.initial_state] = start_hidden_state
            summary = tf.Summary()

            loss, _, adv, v, vl, pl, el = sess.run([nn.loss, nn.train_step, nn.advantage, nn.value_function, nn.value_loss, nn.policy_loss, nn.entropy], feed_dict)
            print(loss, total_episode_reward)
            summary.value.add(tag='Losses/ValueLoss', simple_value=float(vl))
            summary.value.add(tag='Losses/PolicyLoss', simple_value=float(pl))
            summary.value.add(tag='Losses/Entropy', simple_value=float(el))
            summary_writer.add_summary(summary, t)
            summary_writer.flush()

            if t % 1000 == 0:
                saver.save(sess, "training/agent", global_step=t)

            total_trial_reward /= n_episodes_per_trial
            total_trial_rewards.append(total_trial_reward)
            print(total_trial_reward)
            if t%20 == 0:
                plt.figure(2)
                plt.clf()
                plt.plot(total_trial_rewards)
                plt.pause(0.001)
                plt.show()
            open('figreport/trial_rewards.txt', 'w').write(str(total_trial_rewards))

            if t%100 == 0:
                plt.figure(1); plt.savefig('figreport/optimality.png')
                plt.figure(2); plt.savefig('figreport/reward.pdf')
                plt.figure(3); plt.savefig('figreport/pulls.png')


if __name__ == "__main__":
    #  train()
    test()


