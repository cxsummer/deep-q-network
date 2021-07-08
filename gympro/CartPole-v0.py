import gym
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.sequential=nn.Sequential(
            nn.Linear(4,50),
            nn.ReLU(),
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Linear(25,2)
        )
    def forward(self,x):
        return self.sequential(x)

class deepQNet():
    def __init__(self):
        self.γ=0.9
        self.step=0
        self.net=Net()
        self.targetNet=Net()
        self.loss=nn.MSELoss()
        self.optimizer=optim.Adam(self.net.parameters(),lr=0.01)
        self.epsilon=0.1
        self.minepsilon=0.0001
        self.delayStep=2500
        self.memorySize=10000
        self.allStep=100000
        self.batchTrain = 200
        self.replayMemoryStore = deque()
        self.env = gym.make('CartPole-v0').unwrapped  #unwrapped去掉限制，如果不加则会在200步结束

    def selectAction(self,state):
        if self.step<self.delayStep or np.random.uniform()<self.epsilon:
            action=np.random.randint(0,2)
        else:
            state=torch.FloatTensor(state)
            action=self.net(state).argmax().data.item()
        return action

    def executeStep(self,action):
        observation, reward, done, info = self.env.step(action)
        # if done:
        #     reward=-1
        # 加上unwrapped用下面的算法算奖励，可让结果更好
        x, x_dot, theta, theta_dot = observation
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        reward = r1 + r2
        self.step+=1
        if self.step>self.delayStep and self.epsilon>self.minepsilon:
            self.epsilon-=0.00001
        return reward,observation,done

    def saveStore(self,memory):
        self.replayMemoryStore.append(memory)
        if len(self.replayMemoryStore)>self.memorySize:
            self.replayMemoryStore.popleft()

    def experienceReplay(self):
        batchState=[]
        batchAction=[]
        batchReward=[]
        if self.step>self.delayStep and self.step % 100 == 0:
            self.targetNet.load_state_dict(self.net.state_dict())
        batch = random.sample(self.replayMemoryStore,self.batchTrain)
        for item in batch:
            batchState.append(item[0])
            batchAction.append(np.eye(1,2,item[1],dtype=np.int)[0])
            reward=item[2]
            if not item[4]:
                nextState=torch.FloatTensor(item[3])
                reward=torch.FloatTensor([reward])
                reward=reward+self.γ*max(self.targetNet(nextState))
                reward=reward.data.item()
            batchReward.append([reward])
        Q=self.net(torch.FloatTensor(batchState))*torch.FloatTensor(batchAction)
        targetQ=torch.FloatTensor(batchReward)
        self.lossVal=self.loss(targetQ,Q)
        self.lossVal.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train(self):
        state = self.env.reset()
        episode=1
        episodeStep=0
        while self.step< self.allStep:
            self.env.render()
            action=self.selectAction(state)
            reward,nextState,done=self.executeStep(action)
            self.saveStore((state,action,reward,nextState,done))
            episodeStep+=1
            if done:
                state = self.env.reset()
                print("episode:%s,游戏在%s步后结束" %(episode,episodeStep))
                episode+=1
                episodeStep=0
            else:
                state =nextState;

            if self.step>self.delayStep:
                self.experienceReplay()
                if self.step%1000==0:
                    print("step:%s,loss:%s" %(self.step,self.lossVal.data.item()))


dqn=deepQNet()
dqn.train()

