import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.sequential=nn.Sequential(
            nn.Linear(6,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
        self.loss=nn.MSELoss()
        self.optimizer=optim.Adam(self.parameters(),lr=0.001)
    def forward(self, x):
        return self.sequential(x)

class deepQNetWork():
    def __init__(self):
        self.net=Net()
        self.target_net=Net()
        self.γ=0.9
        self.step=0
        self.epsilon=1
        self.decline=0.9
        self.learn_time=0
        self.update_time=20
        self.store_size=2000
        self.b_size=1000
        self.epoises=1000
        self.store=np.zeros((self.store_size,14))

    def select_action(self,state):
        if np.random.uniform()<self.epsilon*(self.decline**self.learn_time):
            action=np.random.randint(0,3)
        else:
            action=self.net(torch.FloatTensor(state)).argmax().item()
        return action

    def execute_action(self,state,action,env):
        next_state, reward, done, info = env.step(action)
        if done:
            reward=10
        index=self.step%self.store_size
        self.store[index][0:6]=state
        self.store[index][6:7]=action
        self.store[index][7:13]=next_state
        self.store[index][13:14]=reward
        return next_state,done

    def experience_replay(self):
        index=np.random.randint(0,self.store_size-self.b_size+1)
        batch=self.store[index:index+self.b_size]
        state=torch.FloatTensor(batch[:,0:6])
        action=torch.LongTensor(batch[:,6:7])
        next_state=torch.FloatTensor(batch[:,7:13])
        reward=torch.FloatTensor(batch[:,13:14])

        Q=self.net(state).gather(1,action)
        target_Q=reward+self.γ*self.target_net(next_state).max(1)[0].reshape(self.b_size,1)
        loss=self.net.loss(Q,target_Q)
        loss.backward()
        self.net.optimizer.step()
        self.net.optimizer.zero_grad()
        self.learn_time+=1
        if self.learn_time%self.update_time==0:
            self.target_net.load_state_dict(self.net.state_dict())

    def train(self):
        env = gym.make('Acrobot-v1').unwrapped
        for i in range(1,self.epoises):
            step=0
            done=False
            state = env.reset()
            while not done:
                step+=1
                self.step+=1
                action=self.select_action(state)
                state,done=self.execute_action(state,action,env)
                env.render()
                if self.step>self.store_size:
                    self.experience_replay()
            print("epoise",i,"step",step)
            if i%10==0:
                self.test(env)
                self.net.train()

    def test(self,env):
        self.net.eval()
        for i in range(10):
            step=0
            done=False
            state = env.reset()
            while not done:
                step+=1
                self.step+=1
                action=self.net(torch.FloatTensor(state)).argmax().item()
                state, reward, done, info = env.step(action)
                env.render()
                if step>1000:
                    print("falil")
                    break
            print("test_step",step)

dqn=deepQNetWork()
dqn.train()






