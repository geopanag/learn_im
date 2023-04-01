

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import glob
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as f




class DeepQNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, device,lr=0.01 ,dropout=0.2 ): 
        super(DeepQNetwork,self).__init__()
        self.input_dims=n_input
        
        self.fc1_dims=n_hidden

        self.fc1 = nn.Linear(self.input_dims , self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims ,1 )
      
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = device 
        self.to(self.device)

    def forward(self, x): 
        x = self.relu(self.fc1(x))
        #x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return x
    



class Agent():
    def __init__(self, gamma, epsilon, lr, state_dim ,action_dim , hidden_dims , batch_size,max_n, max_mem_size,  device, version, eps_end=0.01, eps_dec=5e-4):
        
        self.gamma= gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.state_max = max_n
        self.n_input = state_dim+action_dim
        self.n_hidden = hidden_dims
        self.version = version

        # store the agent's memory position
        self.mem_cnt = 0 
        
        self.Q_eval = DeepQNetwork(self.n_input, self.n_hidden, device,  lr)
        
        self.Q_target = DeepQNetwork(self.n_input, self.n_hidden, device,  lr)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
        self.state_memory = np.zeros((self.mem_size, state_dim ) ,dtype = np.float32) 
        
        
        # each element is the graph state X node embeddings, so the nrows depends on the network size
        self.new_state_memory = np.zeros((self.mem_size, self.state_max , self.n_input  ) ,dtype = np.float32) 
             
        self.action_memory = np.zeros((self.mem_size, action_dim ) ,dtype = np.float32)
        
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)        
        
        #self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        


    def store_transition(self, state, new_state,action,  reward):#feature_idx
        
        # store into memory
        # position of the memory to store, if it is full, start rewriting it
        index = self.mem_cnt % self.mem_size 
        
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cnt+=1

        
        
    def decrement_epsilon(self):
        self.epsilon =self.epsilon - self.eps_dec
        if self.epsilon < self.eps_min:    
            self.epsilon = self.eps_min

        
        
    def learn(self):
        #print(self.mem_cnt )
        if self.mem_cnt < self.batch_size:
            return
        
        self.Q_target.eval()
        
        self.Q_eval.optimizer.zero_grad()
        
        # sample from the memories gathered up to now
        max_mem =  min(self.mem_cnt,self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        
        #batch_index = np.arange(self.batch_size, dtype = np.int32)
        
        
        state_batch = torch.FloatTensor(self.state_memory[batch]).to(self.Q_eval.device)

        
        action_batch = torch.FloatTensor(self.action_memory[batch]).to(self.Q_eval.device)
        
         
        #!!!!!!!!!!!!!!!!!!!  for instead of direct batch with pads
        new_state_batch = torch.FloatTensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        
        reward_batch =torch.FloatTensor(self.reward_memory[batch]).to(self.Q_eval.device)
        
        # cou = 0
        
        # compute the predicted reward
        # instead of indexing [batch_index,action_batch],use directly the representaton of the action seed as input to get the q value     
        q_eval = self.Q_eval.forward(torch.cat([state_batch,action_batch], dim=1)).squeeze()
        
        # compute the estimated reward
        # squeeze out the max_n dimension that corresponds to the nodes 
        q_next = self.Q_target.forward(new_state_batch).squeeze(2)
        #print(q_next.shape)
        
        q_target = reward_batch + self.gamma *torch.max(q_next,dim=1)[0] 
        
        #print(q_target.size())
        #print(q_eval.size())
        
        loss = self.Q_eval.loss(q_eval,q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.decrement_epsilon()
        
        
    

    def save(self):
        torch.save({
                'state_dict':self.Q_eval.state_dict(),
                'optimizer' : self.Q_eval.optimizer.state_dict(),
            }, "../models/model_q"+self.version+'.pth.tar')



