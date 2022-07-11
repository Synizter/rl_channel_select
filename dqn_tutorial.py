from cmath import e
from curses.panel import top_panel
import os
import re
from tkinter import N
from matplotlib import units #models savbing

import numpy as np
import tensorflow.compat.v1 as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.disable_v2_behavior()

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fc1_dims = 256, input_dims=(210, 160,4)
    , chkpt_dir = 'temp/dqn') -> None:
        self.lr = lr
        self.name = name
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variable_initializer())
        self.saver = tf.trian.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)


    def build_net(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape = [None, *self.input_dims],
                                        name = 'inputs')
            
            self.actions = tf.placeholder(tf.float32, shape = [None, self.n_actions],
                                        name = 'actions_taken')

            self.q_target = tf.placeholder(tf.float32, shape = [None, self.n_actions])


            conv1  = tf.layers.conv2d(inputs = self.input, filters = 32, 
                kernel_size= (8,8), strides = 4, name = 'Conv1', 
                kernel_initializer = tf.variance_scaling_initializer(scale = 2))
            conv1_activate = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(inputs = conv1_activate, filters = 64, 
                kernel_size= (4,4),  strides = 2, name = 'Conv2',
                kernel_initializer = tf.variance_scaling_initializer(scale = 2))
            conv2_activation = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(inputs = conv2_activation, filters = 128
                kernel = (3,3), strides = 1, name = 'conv3',
                kernel_initializer = tf.variance_scaling_initializer(scale = 2))
            conv3_activation = tf.nn.relu(conv3)

            flat = tf.layers.flatten(conv3_activation)
            dense1 = tf.layerss.dense(flat, units = self.fc1_dims, activation = tf.nn.relu,
                kernel_initializer = tf.variance_scaling_initializer(scale = 2))

            #State-Action Pair
            self.Q_values = tf.layers.dense(dense1, units = self.n_actions, 
              kernel_initializer = tf.variance_scaling_initializer(scale = 2))
            
            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))
            self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        def load_checkpoint(self):
            print('loading checkpoint.....')
            self.saver.restore(self.sess, self.checkpoint_file)
        
        def save_checkpoint(Self):
            print('saving checkpoint....')
            self.saver.save(self.sess, self.checkpoint_file)
        

class Agent(object):
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size, replace_target=5000,
    input_dims = (210, 160,4), q_next_dir = 'temp/q_next', q_eval_dir = 'temp/q_eval') -> None:
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.mem_size = mem_size
        self.epsilon = epsilon
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.q_next = DeepQNetwork(alpha, n_actions, input_dims, name = 'q_next', chkpt_dir=q_next_dir)
        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims, name = 'q_eval', chkpt_dir=q_next_dir)
        self.state_memory = np.zeors((self.mem_size, *input_dims))
        self.new_state_memory = np.zeors((self.mem_size, *input_dims))
        self.action_memory = np.zeors((self.mem_size, self.n_actions))
        self.reward_memory = np.zeors(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
    
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size 
        self.state_memory[index] = state
        actions = np.zeros(self.actions)
        actions[action] = 1.0

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1
    
    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.q_eval.sess.run(self.q_eval.Q_values, feed_dict = {self.q_eval.input: state})
            action = np.argmax(action)
        return action
    
    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()
        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        batch = np.random.choice(max_mem, self.batch_size)
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        action_values = np.array([0, 1, 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values, feed_dict = {self.q_eval.input: state_batch})
        q_next = self.q_next.sess.run(self.q_next.Q_values, feed_dict = {self.q_next.input: new_state_batch})

        q_target = q_eval.copy()
        q_target[:, action_indices] = reward_batch + \
            self.gamma * np.max(q_next,axis = 1) * terminal_batch
        
        _ = self.q_eval.sess.run(self.q_eval.train_op, feed_dict={
            self.q_eval.input: state_batch,
            self.q_eval.actions: action_batch,
            self.q_eval.q_target: q_target
        })

        if self.mem_cntr > 100000:
            if self.epsilon > 0.01:
                self.epsilon *= 0.9999999
            elif self.epsilon <= 0.01:
                self.epsilon = 0.01
        
    def save_model(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_model(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
    
    def update_graph(self):
        e_param = self.q_eval.params
        t_param = self.q_next.params
    
        for t, e in zip(t_param, e_param):
            self.q_eval.sess.run(tf.assign(t, e))

