"""
Testing ideas from RL^2 : Fast RL via Slow RL (OpenAI) on a multi armed bandit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import agent
import gym


def test():
    with tf.Session() as sess:
        nn = RL2(2)
        saver = tf.train.Saver()
        saver.restore(sess, './training/agent-11000')
        bandit = [0.4, 0.6]
        random.shuffle(bandit)
        choice = random.randint(0,1)
        reward = pull(bandit, choice)
        hidden_state = None
        optimal = []
        
        for i in range(100):
            print(bandit, choice, reward)
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
            optimal.append(choice == np.argmax(bandit))

        print(sum(optimal)/len(optimal))








def train():
    tf.reset_default_graph()
    summary_writer = tf.summary.FileWriter('summaries/v2')

    gravities = [1, 10, 100]

    n_actions = 2
    statesize = 4
    nn = agent.RL2(n_actions, statesize)

    # params
    n_trials = int(1e5)
    n_episodes_per_trial = 3
    max_ep_length = 100
    GAMMA = 0.9

    env = gym.make('CartPole-v0')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)


        for t in range(n_trials):

            env.gravity = random.choice(gravities)
            hidden_state = None

            inputs, stateinputs, rewards, values = [], [], [], []
            episode_lengths = []

            average_trial_reward = 0

            for e in range(n_episodes_per_trial):

                env.reset()
                total_episode_reward = 0

                # taking initial action
                action = random.randint(0, n_actions-1)

                i = 0
                done = False
                while i < max_ep_length and not done:

                    observation, reward, done, info = env.step(action)
                    total_episode_reward += reward

                    stateinputs.append(observation)
                    terminated = 1 if done or i == max_ep_length-1 else 0
                    rnn_input = np.array([action, terminated, reward])
                    inputs.append(rnn_input)
                    rewards.append(reward)

                    feed_dict = { 
                            nn.batch_size : 1,
                            nn.sequence_length: 1,
                            nn.input : [[rnn_input]],
                            nn.state_input: [[observation]],
                    }
                    if hidden_state is not None:
                        feed_dict[nn.initial_state] = hidden_state

                    policy, hidden_state, value = sess.run(
                            [nn.last_actions_distribution, nn.rnn_output_state, nn.value_function], 
                            feed_dict)
                    values.append(value[0][0])

                    # choosing action
                    action = np.random.choice(range(n_actions), p=policy)

                    i += 1

                episode_lengths.append(i)
                average_trial_reward += total_episode_reward

            average_trial_reward /= n_episodes_per_trial

            # TRAINING
            value_plus = values + [0]
            advantages = np.array(rewards) + GAMMA * np.array(value_plus[1:]) - np.array(value_plus[:-1])
            advantages = agent.discount(advantages, GAMMA)

            feed_dict={
                nn.batch_size : 1,
                nn.sequence_length : len(inputs),
                nn.input : np.array([inputs]),
                nn.state_input: [stateinputs],
                nn.reward : [agent.discount(rewards, GAMMA)],
                nn.learning_rate : 1e-3,
                nn.advantage : [advantages],
            }
            summary = tf.Summary()

            loss, _, vl, pl, el = sess.run(
                    [nn.loss, nn.train_step, nn.value_loss, nn.policy_loss, nn.entropy],
                    feed_dict)
            print(loss, total_episode_reward)
            summary.value.add(tag='Losses/Loss', simple_value=float(loss))
            summary.value.add(tag='Losses/ValueLoss', simple_value=float(vl))
            summary.value.add(tag='Losses/PolicyLoss', simple_value=float(pl))
            summary.value.add(tag='Losses/Entropy', simple_value=float(el))
            summary.value.add(tag='Reward/AverageReward', simple_value=average_trial_reward)
            summary_writer.add_summary(summary, t)
            summary_writer.flush()

            if t % 1000 == 0:
                saver.save(sess, "training/agent2", global_step=t)


if __name__ == "__main__":
    train()
    #  test()


