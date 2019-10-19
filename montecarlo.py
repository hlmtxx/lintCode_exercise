import random
import numpy as np

from environment import *

def montecarlo(value,counter):
    state=State()
    totalreward=0
    visits=[]
    while state !="terminal":
        action=None
        e=100.0/(100.0+np.sum(counter[:,state.dealercard,state.playersum],axis=0))
        if (random.random()<e):
            action=random.randint(0,1)
        else:
            action=np.argmax(value[:,state.dealercard,state.playersum])
        counter[action,state.dealercard,state.playersum]+=1
        visits.append((action,state.dealercard,state.playersum))
        state,reward=step(state,action)
        totalreward+=reward
    
    for action,dealercard, playersum in visits:
        a=1/counter[action,dealercard,playersum]
        g=totalreward
        value[action,dealercard,playersum]=value[action,dealercard,playersum]+a*(g-value[action,dealercard,playersum])
    return value,counter