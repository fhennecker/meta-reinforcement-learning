import tensorflow as tf
import numpy as np
import gym
import agent
import matplotlib.pyplot as plt
import random

def test():
    tf.reset_default_graph()

    #  gravities = [1, 10]
    #  pole_lenghts = [0.5, 2, 5]
    #  state_permutations = [
            #  [0, 1, 2, 3]
    #  ]
    gravities = [10]
    pole_lenghts = [0.5]
    state_permutations = [
            #  [0, 1, 2, 3],
            #  [0, 1, 3, 2],
            #  [1, 0, 2, 3]
            [1, 0, 3, 2]
    ]
    state_permutations = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
    ]

    #  state_permutations = [
            #  [2, 1, 3, 0],
            #  [2, 3, 1, 0],
            #  [2, 3, 0, 1],
            #  [3, 1, 2, 0],
            #  [3, 2, 0, 1],
            #  [3, 2, 1, 0]
    #  ]

    #  gravities = [10]
    #  pole_lenghts = [0.5]
    #  state_permutations = [
            #  [0, 1, 2, 3],
            #  [0, 1, 3, 2],
            #  [1, 0, 2, 3]
    #  ]
    #  state_permutations = [[1, 0, 3, 2]]
    #  state_permutations = [
            #  [0, 1, 2, 3],
            #  [1, 2, 3, 0],
            #  [2, 3, 0, 1],
            #  [3, 0, 1, 2],
            #  [1, 0, 3, 2]
    #  ]
    #  state_permutations = [
            #  [0, 3, 1, 2],
            #  [1, 0, 2, 3],
            #  [2, 1, 3, 0],
            #  [3, 2, 1, 0],
            #  [0, 3, 2, 1]
    #  ]



    n_actions = 2
    statesize = 4

    # params
    max_ep_length = 200

    env = gym.make('CartPole-v0')

    number_of_trials = int(len(state_permutations)*4)
    number_of_episodes = [5]
    final_rewards = [[] for i in range(len(number_of_episodes))]



    #  for m, model in enumerate(['../remote/training/20permsLR1epgamma9-450000', '../remote/training/20permsLR2epgamma9-450000']):
    #  for m, model in enumerate(['../remote/training/20permsLR2epgamma9-450000', '../remote/training/20permsLR5epgamma9-360000', '../remote/training/20permsLR10epgamma9-320000']):
    #  for m, model in enumerate(['../remote/training/20permsLR10epgamma9-550000']):
    #  for m, model in enumerate(['../remote/training/3perms10ep-120000']):
    #  for m, model in enumerate(['../remote/training/3perms15ep-170000']):
    for m, model in enumerate(['../remote/training/magic_neg10-300000']):
        print(m, model)
        print()
        tf.reset_default_graph()
        with tf.Session() as sess:
            nn = agent.RL2(2, 4)
            saver = tf.train.Saver()
            saver.restore(sess, model)

            for k in range(number_of_trials):
                for invert in [False, True]:
                    env.gravity = random.choice(gravities)
                    env.length = random.choice(pole_lenghts)
                    env.polemass_length = env.masspole*env.length
                    #  perm = random.choice(state_permutations)
                    perm = state_permutations[k%len(state_permutations)]
                    print("NEW TRIAL", env.gravity, env.length, perm, invert)

                    hidden_state = None

                    final_rewards[m].append([])
                    trained_agent = None

                    for t in range(number_of_episodes[m]):

                        env.reset()
                        action = env.action_space.sample()
                        #  if t == 4:
                            #  trained_agent = hidden_state
                        #  if t > 4:
                            #  hidden_state = trained_agent

                        total_episode_reward = 0
                        for i in range(max_ep_length):
                            inverted_action = action
                            if invert:
                                inverted_action = 0 if action == 1 else 1
                            observation, reward, done, info = env.step(inverted_action)
                            #  env.render()
                            if abs(observation[0]) > 1:
                                done = True
                            observation = np.array(observation)[perm]
                            #  env.render()
                            total_episode_reward += reward
                            terminated = i == max_ep_length-1 or done
                            if terminated:
                                reward = -10

                            feed_dict = { 
                                    nn.batch_size : 1,
                                    nn.sequence_length: 1,
                                    nn.state_input : [[observation]],
                                    nn.input : [[[action, 1 if terminated else 0, reward]]]
                            }
                            if hidden_state is not None:
                                feed_dict[nn.initial_state] = hidden_state

                            actions_distribution, hidden_state = sess.run(
                                    [nn.last_actions_distribution, nn.rnn_output_state], feed_dict)

                            action = np.random.choice(2, p=actions_distribution)
                            if terminated: #done:
                                print("Failed.")
                                break
                            if i == max_ep_length-1:
                                print("Succeeded!")
                        final_rewards[m][-1].append(total_episode_reward)

            plt.figure(figsize=(6, 5))
            plt.ylim([-10, 210])
            plt.plot(np.transpose(np.array(final_rewards[m])), color=(0.28, 0.6, 0.85), alpha=0.15)
            plt.plot(np.mean(np.array(final_rewards[m]), axis=0), 'm--', linewidth=3, label='Average')
            #  plt.boxplot(np.array(final_rewards[m]))
            plt.ylabel('Episode reward')
            plt.xlabel('Episode number within the trial')
            plt.legend( loc='lower right')
            plt.tight_layout()
            plt.savefig(str(m)+'.pdf')
            plt.show()

    #  f1, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[3, 1]})
    #  a0.hist(np.array(final_rewards[0])[:,-1].flatten(), bins=40, range=[0, 200], orientation="horizontal")
    #  a0.invert_xaxis()
    #  a0.set_xlim([180, 0])
    #  a0.set_ylabel('Final episode reward')
    #  a0.set_xlabel('Distribution of episodes')

    #  a1.boxplot(np.array(final_rewards[0])[:,-1].flatten())
    #  a1.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
    #  a1.get_yaxis().set_ticks([])
    #  f1.tight_layout()
    #  f1.savefig("1.pdf")

    #  f2, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1, 3]})
    #  a0.boxplot(np.array(final_rewards[1])[:,-1].flatten())
    #  a0.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
    #  a0.get_yaxis().set_ticks([])

    #  a1.hist(np.array(final_rewards[1])[:,-1].flatten(), bins=40, range=[0, 200], orientation="horizontal")
    #  a1.set_xlim([0, 180])
    #  a1.set_ylabel('Final episode reward')
    #  a1.set_xlabel('Distribution of episodes')
    #  a1.yaxis.tick_right()
    #  a1.yaxis.set_label_position('right')
    #  f2.tight_layout()
    #  f2.savefig("2.pdf")

    plt.show()

if __name__ == "__main__":
    test()


