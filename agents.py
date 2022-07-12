from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import mne_dataset
from processor import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model_set import Custom1DCNN
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


#CONSTANT---------------------------------------------------------
left_ch  = ['C3', 'F3', 'F7', 'FP1', 'O1', 'P3'] #STATES
right_ch  = ['C4', 'F4', 'F8', 'FP2', 'O2', 'P4'] #ACTIONS

ch_depth_2 = list(itertools.product(left_ch, right_ch))
ch3 = list(itertools.product(ch_depth_2, right_ch))
ch4 = list(itertools.product(ch3, right_ch))
ch_depth_3 = []
for i,t in enumerate(ch3):
    x, y = t #a = ch tuple, b = last action added
    a, b = x # c = first ch, d = 2nd ch
    if b != y:
        ch_depth_3.append((a, b, y))

ch_depth_4 = []
for i,t in enumerate(ch4):
    x, y = t #a = tuple of tuple of ch, b = last action added
    c, d = x # c = first ch, d = 2nd ch
    a, b = c
    r = Counter([a, b, d, y])
    if len(r) > 2 and b != d:
        ch_depth_4.append((a, b, d, y))

ch_map = {'FC5': 0, 'FC3': 1, 'FC1': 2, 'FCZ': 3, 'FC2': 4, 'FC4': 5, 'FC6': 6, 'C5': 7,
     'C3': 8, 'C1': 9, 'CZ': 10, 'C2': 11, 'C4': 12, 'C6': 13, 'CP5': 14, 'CP3': 15, 'CP1': 16, 
     'CPZ': 17, 'CP2': 18, 'CP4': 19, 'CP6': 20, 'FP1': 21, 'FPZ': 22, 'FP2': 23, 'AF7': 24, 'AF3': 25, 
     'AFZ': 26, 'AF4': 27, 'AF8': 28, 'F7': 29, 'F5': 30, 'F3': 31, 'F1': 32, 'FZ': 33, 'F2': 34, 'F4': 35, 
     'F6': 36, 'F8': 37, 'FT7': 38, 'FT8': 39, 'T7': 40, 'T8': 41, 'T9': 42, 'T10': 43, 'TP7': 44, 'TP8': 45,
      'P7': 46, 'P5': 47, 'P3': 48, 'P1': 49, 'PZ': 50, 'P2': 51, 'P4': 52, 'P6': 53, 'P8': 54, 'PO7': 55, 
      'PO3': 56, 'POZ': 57, 'PO4': 58, 'PO8': 59, 'O1': 60, 'OZ': 61, 'O2': 62, 'IZ': 63}



actions = right_ch.copy()
states = (left_ch + ch_depth_2 + ch_depth_3 + ch_depth_4).copy()


#QTABLE
q = np.zeros((len(states), len(actions)))


#TRAIN PARAM
learning_decay = 0.99999
alpha = 1e-3 #learning rate
gamma = 0.4 #discount factor
epsilon = 1 #explor/exploit control

nbr_episode = 1000
agent_epochs = 10000

#CLEANUP
del ch3,ch4,left_ch,right_ch,ch_depth_2, ch_depth_3, ch_depth_4
#END CONSTANT-----------------------------------------------------

def qinfo_encode(info):
    '''
    Encode infomation for q table, using index
    to map action (channel to be added) and state (channel combination) 
    using index as key and combination/action as value
    '''
    return {k:v for k, v in enumerate(info)}

def qinfo_decode(index, info):
    return info[index]

def get_ch_location(info):
    '''
    get tuple of ch name in string
    return list of channel location index from ch map
    '''
    if not isinstance(info, tuple):
        return ch_map[info]
    else:
        tmp = []
        for ch in info:
            tmp.append(ch_map[ch])
        return tmp

def extract_target_ch(raws, target_ch:list):
    return raws[:,:,target_ch]


def jump_to_state(state, action):
    if not isinstance(state, tuple):
        target = (state, action)
    else:
        target = (*state, action)
    return list(states_info.keys())[list(states_info.values()).index(target)]

DEBUG = 2
states_info = qinfo_encode(states)
actions_info =  qinfo_encode(actions)

if DEBUG == 2:

    exclude = [38, 88, 89, 92, 100, 104]
    subjs = [s for s in range(1,30 + 1) if s not in exclude]
    #Task 1 and 2, MI vs MM of left and righht fist
    runs = [3,4, 7,8, 13,14]
    raw = load_data(subjs, runs, max_duration = 120)

    x,y = epochs(raw)
    x = normalize(x)
    y = to_one_hot(y[:,-1])
    print(x.shape, y.shape)
    try:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    except ValueError as e:
        print(e)


    # # t = qinfo_decode(1331, states_info)
    # # t = get_ch_location(t)
    # # t = extract_target_ch(raws, t)

    #FORCE USING GPU
    if tf.config.experimental.list_physical_devices('GPU'):
        device = tf.test.gpu_device_name()
    else:
        device = 'CPU:0'

    episode_reward = np.zeros(agent_epochs)
        # reward_trajectory = []
    episode_chs = []            
    for i in range(agent_epochs):
        done = False
        total_reward = 0

        #reset env, randomly select first channel (state 0-5)
        state_index = np.random.randint(0, 6)
        state = qinfo_decode(state_index, states_info)
        
        print("Training on epochs", i)
        while not done:
            
            if np.random.uniform(0, 1) < epsilon:
                action_index = np.random.randint(0, 6)
            else:
                action_index = np.argmax(q[state_index])
            action = actions_info[action_index]
            # Take action ----------------------------------------------------------
            
            #terminate if total ch is larger than 4
            if len((*state, action)) > 4:
                done = True
            elif action in list(state):
                reward = -99
                done = True
            else:
                next_state = 0
                reward = 0
                
                #extract data
                if 0 <= state_index < 6:
                    target_ch = get_ch_location((state, action))
                else: 
                    target_ch = get_ch_location((*state, action))
                temp_x_train = extract_target_ch(x_train, target_ch)
                temp_x_val = extract_target_ch(x_val, target_ch)
                
                # print(temp_x_train.shape, y_train.shape, temp_x_val.shape, y_val.shape)
                
                with tf.device(device):
                    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
                    loss = tf.keras.losses.CategoricalCrossentropy()
                    model = Custom1DCNN(inp_shape=(temp_x_train.shape[1], temp_x_train.shape[2]), output_classes=y_train.shape[1])
                    model.compile(optimizer = opt, loss=loss , metrics=['accuracy'])
                    hist = model.fit(temp_x_train, y_train, 
                                     batch_size= 128, epochs = 50, verbose = False, 
                                     validation_data = (temp_x_val,y_val))
                    testLoss, testAcc = model.evaluate(temp_x_val, y_val)
                
                #get reward and next state after performing an action
                reward = testAcc
                #decode next
                next_state = jump_to_state(state, action) 
                
                #UPDATE Q
                q_value = q[state_index,action_index]
                max_q = np.max(q[next_state])
                
                max_val = np.max(q[next_state])
                new_q  = (1 - alpha) * q_value + alpha * (reward + gamma * max_val)
                q[state_index, action_index] = new_q
                
                state = states_info[next_state]
                state_index = next_state
                
                episode_reward[i] += reward
            #decay the epsilon
            if epsilon > 0.01:
                epsilon *= learning_decay
            else:
                epsilon = epsilon
            # Done taking action ----------------------------------------------------------
        episode_chs.append(state)
        print('accumulate reward of epochs {} with following chs <{}> is {}'.format(i, episode_chs[i], episode_reward[i]))
np.save('episode_reward',episode_reward)
np.save('qtable', q)