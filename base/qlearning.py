import random
import numpy as np
R = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 1],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, -1],
              [-1, 0, -1, -1, -1, -1],
              ])
def qlearning():
    Q=np.zeros([6,6],np.float32)
    α=0.5
    γ=0.8
    for i in range(2000):
        state=random.randint(0,5)
        canAction= [(i,v)[0] for (i,v) in enumerate(R[state]) if (i,v)[1]>-1]
        action=canAction[random.randint(0,len(canAction)-1)]
        Q[state][action]=(1-α)*Q[state][action]+ α*(R[state][action]+γ*max(Q[action]))

    for i in range(5):
        start=i
        end=5
        def getWay(temp,way=''):
            indexs=np.where(Q[temp]==np.max(Q[temp]))[0];
            for index in indexs:
                waytemp=(way+'->'+str(index))
                if index==end:
                    print(waytemp)
                else:
                    getWay(index,waytemp);
        getWay(start,str(start))

qlearning()