import gym
import torch
import random
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.sequential=nn.Sequential(
            nn.Linear(2,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,3),
            nn.Dropout(0.2),
        )
        self.loss=nn.MSELoss()
        self.optimizer=optim.Adam(self.parameters(),lr=0.001)
    def forward(self,x):
        return self.sequential(x)

env = gym.make('MountainCar-v0').unwrapped
net=Net()
net2=Net()
store_count=0
store_size=2000
decline=0.6
learn_time=0
update_time=20
game=0.9
b_size=1000
store=np.zeros((store_size,6))
start_study=False
epoise=0
for i in range(50000):
    s=env.reset()
    step=0
    while True:
        step+=1
        if random.randint(0,100)<100*(decline**learn_time):
            a=random.randint(0,2)
        else:
            out=net(torch.Tensor(s))
            a=torch.argmax(out).data.item()
        s_,r,done,info=env.step(a)
        r = 0
        if s_[0]>0.5:
            r=5
        elif s_[0]>-0.5:
            r=s_[0]+0.5
        store[store_count%store_size][0:2]=s
        store[store_count%store_size][2:3]=a
        store[store_count%store_size][3:5]=s_
        store[store_count%store_size][5:6]=r
        store_count+=1
        s=s_
        if store_count>store_size:
            if learn_time%update_time==0:
                net2.load_state_dict(net.state_dict())
            index=random.randint(0,store_size-b_size-1)

            b_s=torch.Tensor(store[index:index+b_size,0:2])
            b_a=torch.LongTensor(store[index:index+b_size,2:3])
            b_s_=torch.Tensor(store[index:index+b_size,3:5])
            b_r=torch.Tensor(store[index:index+b_size,5:6])
            q=net(b_s).gather(1,b_a)
            q_next=net2(b_s_).max(1)[0].reshape(b_size,1)
            tq=b_r+game*q_next
            loss=net.loss(q,tq)
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()

            learn_time+=1
            if not start_study:
                print("start_study")
                start_study=True
                break
        if done:
            print("epoise",i,"step",step)
            break
        env.render()




