from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras import backend as K 
import numpy as np 


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        state = self.state_memory[batch]
        state_ = self.new_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        done = self.terminal_memory[batch]
        return state, action, reward, state_, done 


def build_neural_network(learning_rate, input_dims, n_actions, fc1):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, input_shape=(*input_dims,), strides=4, activation='relu'
                     , data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fc1, activation='relu'))
    model.add(Dense(n_actions))
    model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')
    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace,
                 input_dims, eps_decay=1e-5, eps_min=0.01, mem_size=1000000,
                 q_eval_fname='keras-deep-q-pong/models/q_eval.h5', q_target_fname='keras-deep-q-pong/models/q_target.h5'):
        self.alpha = alpha 
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma 
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.q_eval_fname = q_eval_fname
        self.q_target_fname = q_target_fname
        self.learn_step = 0 
        self.memory = ReplayBuffer(mem_size, input_shape=input_dims)
        self.q_eval = build_neural_network(learning_rate=alpha, input_dims=input_dims,
                                           n_actions=n_actions, fc1=512)
        self.q_next = build_neural_network(learning_rate=alpha, input_dims=input_dims, 
                                           n_actions= n_actions, fc1=512)
    
    def replace_target_network(self):
        if self.replace != 0 and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def choose_action(self, observation):
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], dtype=np.float32, copy=False)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)
            self.replace_target_network()
            
            q_eval = self.q_eval.predict(state)
            q_next = self.q_next.predict(state_)
            
            q_next[done] = 0.0
            indices = np.arange(self.batch_size)
            q_target = q_eval[:]
            
            q_target[indices, action] = reward + self.gamma * np.max(q_next, axis=1)
            
            self.q_eval.train_on_batch(state, q_target)
            self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_min \
                else self.eps_min
            self.learn_step += 1
            
    def save_models(self):
        self.q_eval.save(self.q_eval_fname)
        self.q_next.save(self.q_target_fname)
        print("Models Saved")
    
    def load_models(self):
        self.q_eval = load_model(self.q_eval_fname)
        self.q_next = load_model(self.q_target_fname)
        print("Models Loaded")
    

    
            
        
        
        
    
    
        

