import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.utils.data as Data

R = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 1],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, -1],
              [-1, 0, -1, -1, -1, -1],
              ])

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.sequential=nn.Sequential(
            nn.Linear(6,8),
            nn.ReLU(),
            nn.Linear(8,6),
        )
    def forward(self,x):
        return self.sequential(x)

class deepQNet():
    def __init__(self):
        self.γ=0.9
        self.step=0
        self.net=Net()
        self.loss=nn.MSELoss()
        self.optimer=optim.SGD(self.net.parameters(),lr=0.1)
        self.epsilon=0.1
        self.minepsilon=0.0001
        self.delayStep = 2500
        self.memorySize=5000
        self.allStep=10000
        self.batchTrain = 200
        self.replayMemoryStore = deque()

    def selectAction(self,state):
        if self.step<self.delayStep or np.random.uniform() < self.epsilon :
            currentAction = np.random.randint(0, 6)
        else:
            oneHotState=torch.zeros(1,6).scatter_(1,torch.tensor([[state]]),1)[0]
            currentAction = self.net(oneHotState).argmax()
        return currentAction

    def executeStep(self, state, action):
        reward=R[state][action]
        nextState=action
        done=nextState==5
        self.step+=1
        if self.step>self.delayStep and self.epsilon>self.minepsilon:
            self.epsilon-=0.00001
        return state,action,reward,nextState,done

    def saveStore(self,memory):
        self.replayMemoryStore.append(memory)
        if len(self.replayMemoryStore)>self.memorySize:
            self.replayMemoryStore.popleft()

    def experienceReplay(self):
        batchState=[]
        batchAction=[]
        batchReward=[]
        dataSet=Data.TensorDataset(torch.tensor(self.replayMemoryStore) )
        dataLoader=Data.DataLoader(dataSet,self.batchTrain,shuffle=True)
        for item in next(dataLoader.__iter__())[0]:
            state=torch.zeros(1,6).scatter_(1,torch.tensor([[item[0]]]),1)[0]
            action=torch.zeros(1,6).scatter_(1,torch.tensor([[item[1]]]),1)[0]
            nextState=torch.zeros(1,6).scatter_(1,torch.tensor([[item[3]]]),1)[0]
            reward=item[2]
            if reward>=0:
                reward=reward+self.γ*max(self.net(nextState))
            batchState.append(state.data.numpy())
            batchAction.append(action.data.numpy())
            batchReward.append([reward.data.item()])
        Q=self.net(torch.tensor(batchState,dtype=torch.float32))*torch.tensor(batchAction,dtype=torch.float32)
        targetQ=torch.tensor(batchReward,dtype=torch.float32)
        self.lossVal= self.loss(Q,targetQ)
        self.lossVal.backward()
        self.optimer.step()
        self.optimer.zero_grad()

    def train(self):
        state=random.randint(0,4)
        while self.step<=self.allStep:
            action= self.selectAction(state)
            state,action,reward,nextState,done=self.executeStep(state,action)
            self.saveStore((state,action,reward,nextState,done))
            if done:
                state=random.randint(0,4)
            else:
                state=nextState
            if self.step>self.delayStep:
                self.experienceReplay()
                if self.step%1000==0:
                    print("step:%d,loss:%s" %(self.step,self.lossVal.data.item()))

    def test(self):
        for i in range(6):
            start=i
            print("从%s开始走" %(start))
            step=0
            while start!=5:
                state=torch.zeros(1,6).scatter_(1,torch.tensor([[start]]),1)[0]
                start=self.net(torch.tensor(state)).argmax().data.item()
                step+=1
                if start!=5:
                    print("经过%s房间" %(start))
                if step>10:
                    print("失败")
                    break
            print("到达%s房间" %(start))
            print()

dqn=deepQNet()
dqn.train()
dqn.test()





