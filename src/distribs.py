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

    state_permutations = [
            [2, 1, 3, 0],
            [2, 3, 1, 0],
            [2, 3, 0, 1],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0]
    ]

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

    final_rewards = [[], []]
    number_of_trials = int(len(state_permutations)*30)
    number_of_episodes = [1, 2]



    for m, model in enumerate(['../remote/training/20perms1ep-490000', '../remote/training/20perms2ep-490000']):
        print(m, model)
        print()
        tf.reset_default_graph()
        with tf.Session() as sess:
            nn = agent.RL2(2, 4)
            saver = tf.train.Saver()
            saver.restore(sess, model)

            for k in range(number_of_trials):
                for invert in [False]:
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
                            observation = np.array(observation)[perm]
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
                        final_rewards[m][-1].append(total_episode_reward)

    #  plt.ylim([-10, 210])
    #  plt.plot(np.transpose(np.array(final_rewards)), color=(1, 0, 0), alpha=0.4)
    #  plt.plot(np.mean(np.array(final_rewards), axis=0), linewidth=2)
    #  plt.show()
    f1, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[3, 1]})
    a0.hist(np.array(final_rewards[0])[:,-1].flatten(), bins=40, range=[0, 200], orientation="horizontal")
    a0.invert_xaxis()
    a0.set_ylabel('Final episode reward')
    a0.set_xlabel('Distribution of episodes')

    a1.boxplot(np.array(final_rewards[0])[:,-1].flatten())
    a1.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
    a1.get_yaxis().set_ticks([])
    f1.tight_layout()
    f1.savefig("1.pdf")

    f2, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1, 3]})
    a0.boxplot(np.array(final_rewards[1])[:,-1].flatten())
    a0.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
    a0.get_yaxis().set_ticks([])

    a1.hist(np.array(final_rewards[1])[:,-1].flatten(), bins=40, range=[0, 200], orientation="horizontal")
    a1.set_ylabel('Final episode reward')
    a1.set_xlabel('Distribution of episodes')
    a1.yaxis.tick_right()
    a1.yaxis.set_label_position('right')
    f2.tight_layout()
    f2.savefig("2.pdf")

    plt.show()

if __name__ == "__main__":
    test()


