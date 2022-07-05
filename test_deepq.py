'''
DQN
Input : current chs set (eg. i = [1 1 1 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0])
i represent an current channel being used (indicated by 1) and map
the action wheter to add which channel

Output: Channel to be added (eg. o = [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0])
o can only contain best candidate of ch to be add in order to change the current state
activation should be softmax

1. implement
2. *** if the output contain the one in the class, tackle this
'''
# import tensorflow as tf

# class DQN(object):
#        def __init__(self, n_action = 19, n_state = 19, hidden = 1024, learning_rate = 1e-4, history_length = 4):
#               self.n_action = n_action
#               self.n_state= n_state
#               self.learning_rate = learning_rate
#               self.history_length = history_length
              
#               self.dense1 = tf.keras.layers.Dense(units)


import numpy as np
import random
import gym
import itertools

def train_q(env, train_episode, al = .1, gm = .6, eps = .1, render = True, interval = 10):
       q_table = np.zeros([env.observation_space.n, env.action_space.n])
       
       for eps in range(train_episode):
              state = env.reset()
              r = 0
              done = False
              i = 0
              env.render()
              while not done:
                     if random.uniform(0, 1) < eps:
                            action = env.action_space.sample()
                     else:
                            action = np.argmax(q_table[state])
                     
                     next_state, r, done, info = env.step(action)
                     
                     # recalculate
                     old_q = q_table[state, action]
                     max_q = np.max(q_table[next_state])
                     new_q = (1-al) * old_q + al * (r + gm * max_q)
                     
                     #update
                     q_table[state, action] = new_q
                     state = next_state
                     
              print("Episode : {} end".format(eps + 1))   
              
              if render:
                     if (eps + 1) % interval == 0:
                            env.render()
       
       return q_table
def eval_q(env, q_table, eval_episode = 1000):
       panelties_list = []
       epochs_list = []
       print("Start q table evaluation")
       for eval_ep in range(eval_episode):
              state = env.reset()
              epochs = 0
              panelties = 0
              reward = 0
              
              done = False
              
              while not done:
                     action = np.argmax(q_table[state])
                     state, reward, done, _ = env.step(action)
                     #count panalties in single run
                     if reward == -10:
                            panelties += 1 
                            print("Current panelties of eval episode {} = {}".format(eval_ep, panelties))
                     epochs += 1
              panelties_list.append(panelties)
              epochs_list.append(epochs)
       print("end q table evaluation")
       return np.vstack([np.array(panelties_list), np.array(epochs_list)])
def test_qlearn_gym():
       env = gym.make('Taxi-v3').env
       q_table = train_q(env, int(1e7))
       np.save('q_table.npy', q_table)
       np.load('q_table.npy')
       print(q_table)
       # perf = eval_q(env, q_table)

       # print(perf)


def init_ch_q():
       excluded_ch=['Fp1', 'Fp2', 'F7', 'F8', 
                     'Fz', 'F3', 'F4', 'O1', 'O2']
       chs = ['F4', 'C4', 'P4', 'Cz', 
               'F3', 'C3', 'P3', 'F7', 
               'T3', 'T5', 'Fp1', 'Fp2', 
               'T4', 'F8', 'Fz', 'Pz', 'T6', 
               'O2', 'O1']
       
       state_ch = [ch for ch in chs if ch not in excluded_ch]
       print(state_ch)

       #create combination of channle
       obj_iter = itertools.combinations(state_ch, 3)
       observation = []
       for obj in obj_iter:
              observation.append(obj)
       
       action = excluded_ch.copy()
       
       return np.zeros((shape))
              

init_ch_q()
       