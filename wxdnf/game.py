from wxdnf import adb_handler as adb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(12, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 24, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Linear(100, 2),
            nn.ReLU()
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.sequential(x)


def get_state():
    state = []
    for i in range(4):
        adb.get_screen(i)
        # img_pil = Image.open('/Users/user/Documents/MuMu共享文件夹/screen' + str(i) + '.jpg')
        # img_pil_1 = np.array(img_pil)
        # state.append(img_pil_1)
        # print(img_pil_1.shape)
    return


def step(a):
    adb.click_handle(a[0], a[1])
    return get_state()


if __name__ == '__main__':
    #get_state()
    print(123//10)
    print(123%10)
