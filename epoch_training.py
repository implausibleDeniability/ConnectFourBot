from multiprocessing import Pool

import numpy as np
import torch

from kaggle_environments import make

from mcts import mcts
from agents import processObservation


def getdata(agent, against, num, num_workers):
    
    if num_workers > 1:
        print("here")
        with Pool(num_workers) as p:
            print([[agent, against, num]] * num_workers)
            training_data = p.map(
                selfplay, [[agent, against, num]] * num_workers)
        data = training_data[0]
        for i in range(num - 1):
            data += training_data[i+1]
    else:
        data = selfplay(agent, against, num)
    
    return data


def selfplay(agent, against, num):
    training_data = []
    for game in range(num):
        turn = np.random.randint(2)
        env = make("connectx")
        if turn == 0:
            trainer = env.train([None, against])
        else:
            trainer = env.train([against, None])
        done = False
        states = []
        observation = trainer.reset()

        root = None
        while not done:
            policy = mcts(observation, env.state,
                                agent, against)
            # print(observation)
            action = int(np.random.choice(range(7), p=policy))
            states.append([processObservation(observation), policy])
            # print(turn, processObservation(observation).reshape(6, 7), policy, sep='\n')
            observation, reward, done, info = trainer.step(action)
            if reward == None:
                reward = -1
        print("|", end='')
        training_data.append({'states': states, 'result': reward})
    return training_data


def net_update(model, training_data, optimizer, reduction='mean'): 
    loss = 0
    optimizer.zero_grad()
    number_of_states = 0
    for example in training_data:
        for state in example['states']:
            netinput = torch.tensor(state[0], dtype=torch.float32)
            netoutput = model(netinput)
            loss += (netoutput[1] - example['result']) ** 2
            loss -= torch.sum(torch.tensor(state[1], dtype=torch.float32)
                              * torch.nn.functional.log_softmax(netoutput[0], 0))
            number_of_states += 1
    if (reduction == 'mean'):
        loss /= number_of_states
    loss.backward()
    optimizer.step()
