import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


# env = gym.make('BipedalWalkerHardcore-v3').unwrapped
# for i in range(1000):
#     env.render()
#     a = -1+2*np.random.random((1,4))[0]
#     observation, reward, done, info= env.step(a)
#     print(reward)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return 2*self.sequential(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(4, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class deepQNetWork():
    def __init__(self):
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.step = 0
        self.γ = 0.99
        self.tau = 0.02
        self.epoises = 1000
        self.store_size = 10000
        self.b_size = 32
        self.update_time = 20
        self.noise = 0.1
        self.epsilon = 0.1
        self.replay_memory_store = np.zeros((self.store_size, 3 + 1 + 1 + 3 + 1))
        self.env = gym.make('Pendulum-v0').unwrapped

    def selectAction(self, state):
        action = self.actor(torch.FloatTensor(state)).data.numpy()
        return action

    def excuteAction(self, state, action):
        index = self.step % self.store_size
        observation, reward, done, info = self.env.step(action)
        self.replay_memory_store[index][0:3] = state
        self.replay_memory_store[index][3:4] = action
        self.replay_memory_store[index][4:5] = reward
        self.replay_memory_store[index][5:8] = observation
        self.replay_memory_store[index][8:9] = done
        self.step += 1
        return observation, reward, done

    def experience_replay(self):
        if self.step < self.b_size:
            return
        all_num = self.step if self.step < self.store_size else self.store_size
        index = np.random.randint(0, all_num - self.b_size + 1)
        batch = self.replay_memory_store[index:index + self.b_size]
        state = torch.FloatTensor(batch[:, 0:3])
        action = torch.FloatTensor(batch[:, 3:4])
        reward = torch.FloatTensor(batch[:, 4:5])
        observation = torch.FloatTensor(batch[:, 5:8])
        done = torch.FloatTensor(1 - batch[:, 8:9])

        next_action = self.actor_target(observation)
        target_Q = reward + done * self.γ * self.critic_target(observation, next_action).detach()
        Q = self.critic(state, action)
        loss = self.critic.loss(Q, target_Q)
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()

        loss_actor = -torch.mean(self.critic(state, self.actor(state)))
        self.actor.optimizer.zero_grad()
        loss_actor.backward()
        self.actor.optimizer.step()
        if self.step % self.update_time == 0:
            for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)
            for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)

    def train(self):
        for i in range(self.epoises):
            step = 0
            rewards = 0
            state = self.env.reset()
            for s in range(500):
                action = self.selectAction(state)
                state, reward, done = self.excuteAction(state, action)
                self.env.render()
                step += 1
                rewards += reward
                self.experience_replay()
            print("epoise:", i, "step", step, "rewards", rewards)


ddpg = deepQNetWork()
ddpg.train()
