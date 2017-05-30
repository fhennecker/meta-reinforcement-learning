import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import sys
import glob

def mavg(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

files = ['../remote/res_perms_hi.txt', '../remote/res_perms_lo.txt', 
        '../remote/res_params_hi.txt', '../remote/res_params_lo.txt',
        '../remote/res_perms_lo2.txt', '../remote/res_perms_hi_3ep.txt',
        '../remote/res_params_lo_3ep.txt']
files = ['../remote/res_perms_gamma99.txt', '../remote/res_4perms.txt',
        '../remote/res_4perms_lo.txt']
files = ['../remote/res_4perms95.txt', '../remote/res_4perms99.txt',
        '../remote/res_perms_gamma95.txt', '../remote/res_perms_gamma99.txt']
files = glob.glob('../remote/res_perms5ep*.txt')
#  files = ['../remote/res_perms_1ep.txt', '../remote/res_perms_2ep.txt', '../remote/res_5perms_1ep.txt', '../remote/res_5perms_2ep.txt']
#  files = ['../remote/res_20perms_1ep.txt', '../remote/res_20perms_2ep.txt']
#  files = ['../remote/res_20permsLR_1ep.txt', '../remote/res_20permsLR_2ep.txt']
#  files = glob.glob('../remote/res*20*LR*ep*gamma*.txt')
#  files = ['../remote/res_20perms_2ep.txt']# '../remote/res_20perms_2ep.txt']
#  files = ['../remote/res_20permsLR_5ep_gamma9.txt']# '../remote/res_20perms_2ep.txt']

files = ['final_reward.txt']
files = ['../remote/res_20permsLR_10ep_gamma9_from320K.txt', '../remote/res_20permsLR_15ep_gamma9.txt']
files = ['../remote/res_20permsLR_10ep_gamma9_from540K.txt', '../remote/res_20permsLR_15ep_gamma9_from180K.txt']
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 's' :
        files = glob.glob('../remote/res_3perms1*txt')
    if len(sys.argv) > 1 and sys.argv[1] == 'm' :
        files = ['../remote/res_magic_neg10.txt', '../remote/res_magic_good_neg.txt', '../remote/res_magic_neg.txt']
    for filename in files:
        with open(filename, 'r') as f:
            print(filename)
            lines = f.readlines()
            if filename.endswith('from540K.txt'):
                prevlines = open('../remote/res_20permsLR_10ep_gamma9.txt', 'r').readlines()[:320000]
                prevlines2 = open('../remote/res_20permsLR_10ep_gamma9_from320K.txt', 'r').readlines()[:220000]
                lines = prevlines + prevlines2 + lines
            if filename.endswith('from180K.txt'):
                prevlines = open('../remote/res_20permsLR_15ep_gamma9.txt', 'r').readlines()[:180000]
                lines = prevlines + lines

            get_losses = lambda l: eval(l.strip().split(':')[4])[:4]
            get_rewards = lambda l: eval(l.strip().split(':')[4])[4:]


            def reward_per_episode(smoothing=20):

                rewards = np.array(list(map(get_rewards, lines)))

                smoothed_rewards = np.zeros((rewards.shape[0]-smoothing+1, rewards.shape[1]))
                mins = np.zeros((rewards.shape[0]-smoothing+1, rewards.shape[1]))
                for episode in range(rewards.shape[1]):
                    smoothed_rewards[:, episode] = mavg(rewards[:, episode], smoothing)
                    #  for t in range(rewards.shape[0]-smoothing+1):
                        #  mins[t, episode] = np.min(rewards[t:t+smoothing, episode])

                plt.figure(1, figsize=(11, 6))
                ax = plt.subplot(111)
                colors = [plt.get_cmap('Blues')(1.*(i+1)/(1+rewards.shape[1])) for i in range(rewards.shape[1])]
                ax.set_prop_cycle(cycler('color', colors))
                ax.set_ylim([-10, 210])
                ax.plot(smoothed_rewards)
                #  plt.plot(mavg(np.mean(rewards, 1), smoothing), lw=2, color='m')

                #  colors = [plt.get_cmap('Blues')(1.*(i+1)/(1+rewards.shape[1])) for i in range(rewards.shape[1])]
                #  plt.rc('axes', prop_cycle=cycler('color', colors))
                #  plt.plot(mins, ':')
                #  plt.savefig(filename+'.png')
                ax.set_ylabel('Smoothed episode reward')
                ax.set_xlabel('Trial')
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(['Episode'+str(i+1) for i in range(20)], bbox_to_anchor=(1, 0.5), loc='center left')
                plt.savefig(filename+'.pdf')
                plt.show()

            if len(sys.argv) > 1 and sys.argv[1] == 's':
                reward_per_episode(2000)
            else:
                reward_per_episode(2000)
