import numpy as np

class QLearnAgent(object):
    def __init__(self, lr, gamma, n_actions, n_states, epsilon, decay = 0.999):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.decay = decay

        self.Q = np.zeros((n_actions, n_states))
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = self.Q[np.random.choice()]
        else:
            action = np.argmax(Q[state])
        return action


    def epsilon_decay(self):
        self.epsilon *= self.decay if self.epsilon > 0.01 else self.epsilon

    
    def learn(self, state, action, reward, next_state, step_fnc):
        action = self.choose_action(state)
        
        #Train model

        reward, next_state = step_fnc(action)

        self.Q[(state, action)] += self.lr * (reward + self.gamma * self.Q[(next_state, action)])



import itertools
from collections import Counter

chs = chs = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'OZ', 'P3', 'P4', 'PZ']
'''
LEFT    : C3 F3 F7 FP1 O1 P3 
CENTER  : CZ OZ PZ FZ
RIGHT   : C4 F4 F8 FP2 O2 P4

state = left ch
action = right ch

terminate if ch_len > 4, select the same channel in list
each episode get
1. reward
'''

#GENERATE Q TABLE with ENCODED INFORMATION
left_ch  = ['C3', 'F3', 'F7', 'FP1', 'O1', 'P3'] #STATES
right_ch  = ['C4', 'F4', 'F8', 'FP2', 'O2', 'P4'] #ACTIONS

ch_depth_2 = list(itertools.product(left_ch, right_ch))
ch3 = list(itertools.product(ch_depth_2, right_ch))
ch4 = list(itertools.product(ch3, right_ch))

ch_depth_3 = []
for i,t in enumerate(ch3):
    x, y = t #a = ch tuple, b = last action added
    a, b = x # c = first ch, d = 2nd ch
    ch_depth_3.append((a, b, y))

ch_depth_4 = []
for i,t in enumerate(ch4):
    x, y = t #a = tuple of tuple of ch, b = last action added
    c, d = x # c = first ch, d = 2nd ch
    a, b = c
    r = Counter([a, b, d, y])
    if len(r) > 2 and b != d:
        ch_depth_4.append((a, b, d, y))

nbr_actions = len(right_ch)
nbr_states = len(ch_depth_2) + len(ch_depth_3) + len(ch_depth_4)
q = np.zeros((nbr_states, nbr_actions))
print(q.shape)
