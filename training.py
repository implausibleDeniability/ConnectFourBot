import torch
from torch import nn
import numpy as np
from kaggle_environments import make

from model import SimpleNet
from mcts import mcts
from agents import netAgent, processObservation


model = SimpleNet(42, 7)
defaultModel = SimpleNet(42, 7)


optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)
for epoch in range(10):
    training_data = []
    agent = netAgent(model, return_probs=True)
    against = netAgent(defaultModel, incorrect_moves=False)

    for game in range(4):
        turn = np.random.randint(2)
        env = make("connectx")
        if turn == 0:
            trainer = env.train([None, against])
        else:
            trainer = env.train([against, None])
        done = False
        states = []
        observation = trainer.reset()
        while not done:
            policy = mcts(observation, env.state, agent, against)
            states.append([processObservation(observation), policy])
            action = int(np.random.choice(range(7), p=policy))
            observation, reward, done, info = trainer.step(action)
            if reward == None:
                reward = -1
        print("Turn: ", turn + 1)
        print(np.reshape(observation['board'], (6, 7)), end='\n\n')
        training_data.append({'states':states, 'result':reward})
    
    loss = 0
    optimizer.zero_grad()
    for example in training_data:
        for state in example['states']:
            netinput = torch.tensor(state[0], dtype=torch.float32)
            netoutput = model(netinput)
            loss += (netoutput[1] - example['result']) ** 2
            loss -= torch.sum(torch.tensor(state[1], dtype=torch.float32) * torch.nn.functional.log_softmax(netoutput[0], 0))
    loss.backward()
    optimizer.step()
    
    agent = netAgent(model, incorrect_moves=False, best_move=False)
    against = netAgent(defaultModel, incorrect_moves=False, best_move=False)
    result = 0
    for i in range(100):
        env = make('connectx')
        laststate = env.run([agent, against])[-1]
        if laststate[0]['reward'] is None:
            gamereward = -1
        elif laststate[1]['reward'] is None:
            gamereward = 1
        else:
            gamereward = laststate[0]['reward']
        result += (gamereward + 1) / 2
        print(gamereward + 1, end='')

        laststate = env.run([against, agent])[-1]
        if laststate[0]['reward'] is None:
            gamereward = 1
        elif laststate[1]['reward'] is None:
            gamereward = -1
        else:
            gamereward = laststate[1]['reward']
        result += (gamereward + 1) / 2
        print(gamereward + 1, end='')
    
    print("Test result: ", result)
    if (result > 140):
        defaultModel.load_state_dict(model.state_dict())
        print("switch")


torch.save(model.state_dict(), "parameters_simple.pth")