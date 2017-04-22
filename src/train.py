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
    tf.reset_default_graph()

    gravities = [1, 10]
    pole_lenghts = [0.5, 2, 5]
    #  state_permutations = [
            #  [0, 1, 2, 3],
            #  [0, 1, 3, 2],
            #  [1, 0, 2, 3]
    #  ]
    #  state_permutations = [
            #  [0, 1, 2, 3]
    #  ]

    n_actions = 2
    statesize = 4

    # params
    max_ep_length = 200
    GAMMA = 0.9

    env = gym.make('CartPole-v0')

    final_rewards = []


    with tf.Session() as sess:
        nn = agent.RL2(2, 4)
        saver = tf.train.Saver()
        saver.restore(sess, './training/grav_lens_long-99000')

        for k in range(50):
            env.gravity = random.choice(gravities)
            env.length = random.choice(pole_lenghts)
            env.polemass_length = env.masspole*env.length
            #  perm = random.choice(state_permutations)
            print("NEW TRIAL", env.gravity, env.length)

            hidden_state = None

            final_rewards.append([])
            trained_agent = None

            for t in range(20):

                env.reset()
                action = env.action_space.sample()
                #  if t == 4:
                    #  trained_agent = hidden_state
                #  if t > 4:
                    #  hidden_state = trained_agent

                total_episode_reward = 0
                for i in range(max_ep_length):
                    observation, reward, done, info = env.step(action)
                    #  observation = np.array(observation)[perm]
                    #  env.render()
                    total_episode_reward += reward

                    feed_dict = { 
                            nn.batch_size : 1,
                            nn.sequence_length: 1,
                            nn.state_input : [[observation]],
                            nn.input : [[[action, 1 if done else 0, reward]]]
                    }
                    if hidden_state is not None:
                        feed_dict[nn.initial_state] = hidden_state

                    actions_distribution, hidden_state = sess.run(
                            [nn.last_actions_distribution, nn.rnn_output_state], feed_dict)

                    action = np.random.choice(2, p=actions_distribution)
                    if done:
                        print("Failed.")
                        break
                    if i == max_ep_length-1:
                        print("Succeeded!")
                final_rewards[-1].append(total_episode_reward)

        plt.ylim([-10, 210])
        plt.plot(np.transpose(np.array(final_rewards)), color=(1, 0, 0), alpha=0.4)
        plt.plot(np.mean(np.array(final_rewards), axis=0), linewidth=2)
        plt.show()




def train():
    tf.reset_default_graph()
    #  summary_writer = tf.summary.FileWriter('summaries/perms')

    gravities = [10]
    pole_lenghts = [0.5]
    state_permutations = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [1, 0, 2, 3]
    ]

    n_actions = 2
    statesize = 4
    nn = agent.RL2(n_actions, statesize)

    # params
    n_trials = int(1e5)
    n_episodes_per_trial = 5
    max_ep_length = 200
    GAMMA = 0.9

    env = gym.make('CartPole-v0')

    results_file = open('results.txt', 'w')

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=100)
        init = tf.global_variables_initializer()
        sess.run(init)


        for t in range(n_trials):

            gravity = random.choice(gravities)
            length = random.choice(pole_lenghts)
            env.gravity = gravity
            env.length = length
            env.polemass_length = env.masspole*env.length
            perm = random.choice(state_permutations)
            hidden_state = None

            inputs, stateinputs, rewards, values = [], [], [], []
            episode_lengths = []

            trial_rewards = []

            for e in range(n_episodes_per_trial):

                env.reset()
                total_episode_reward = 0

                # taking initial action
                action = random.randint(0, n_actions-1)

                i = 0
                done = False
                while i < max_ep_length and not done:

                    observation, reward, done, info = env.step(action)
                    observation = np.array(observation)[perm]
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
                trial_rewards.append(total_episode_reward)


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
            results_file.write(
                    str(t)+':'+str(gravity)+':'+str(length)+':'+str(perm)+':'+
                    str([loss, vl, pl, el] + trial_rewards)+'\n')
            #  summary.value.add(tag='Losses/Loss', simple_value=float(loss))
            #  summary.value.add(tag='Losses/ValueLoss', simple_value=float(vl))
            #  summary.value.add(tag='Losses/PolicyLoss', simple_value=float(pl))
            #  summary.value.add(tag='Losses/Entropy', simple_value=float(el))
            #  summary.value.add(tag='Reward/AverageReward', simple_value=average_trial_reward)
            #  summary_writer.add_summary(summary, t)
            #  summary_writer.flush()

            if t % 1000 == 0:
                saver.save(sess, "training/null", global_step=t)
        results_file.close()


if __name__ == "__main__":
    train()
    #  test()
