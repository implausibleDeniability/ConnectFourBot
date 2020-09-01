from model import SimpleNet
import torch

net = SimpleNet(42, 7)
net.load_state_dict(torch.load('parameters_simple.pth'))
file = open("submission2.py", 'w')

string = f"""
from agents import netAgent, processObservation
from mcts import mcts
from model import SimpleNet
from torch import nn
from kaggle_environments import make
import numpy as np
import torch
from collections import OrderedDict

def processObservation(observation):
    board = np.array(observation['board'])
    if (observation['mark'] == 2):
        ones = np.argwhere(board == 1)
        twos = np.argwhere(board == 2)
        board[ones] = 2
        board[twos] = 1
    return board


def netAgent(network, return_probs=False, incorrect_moves=True):
    if return_probs:
        def agent(observation, configuration=None):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            output = network(input)
            probs = torch.softmax(output[0], 0)
            return probs, output[1]
    else:
        def agent(observation, configuration=None, incorrect_moves=incorrect_moves):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            output = network(input)
            probs = torch.softmax(output[0], 0)
            if incorrect_moves:
                return torch.argmax(probs).item()
            else:
                action = torch.argmax(probs).item()
                while board[action] != 0:
                    probs[action] = -1
                    action = torch.argmax(probs).item()
                return action
    return agent


ACTIONS_N = 7
U_COEF = 4
MCTS_WAVES = 300
TEMPERATURE = 0.5

CURRENT_N = 1


class Node:
    def __init__(self, value, probs, done=False):
        self.value = value
        self.probs = probs
        self.number = 1
        self.done = done
        self.links = [None] * ACTIONS_N


def mcts(observation, state, agent, against):
    probs, value = agent(observation)
    root = Node(value, probs)
    current_n = 0

    for i in range(MCTS_WAVES):
        current_n += 1
        env = make("connectx", debug=False)
        if (observation['mark'] == 1):
            trainer = env.train([None, against])
        elif (observation['mark'] == 2):
            trainer = env.train([against, None])
        else:
            print("problem with order")
        env.state = state.copy()
        wave(root, agent, trainer, current_n)

    policy = []
    for child in root.links:
        if child is None:
            policy.append(0)
        else:
            policy.append(child.number)
    print(policy)
    policy = np.array(policy)
    policy = policy ** (1 / TEMPERATURE)
    policy = policy / policy.sum()

    return policy


def wave(node, agent, trainer, wave_n):
    if (node.done):
        node.number += 1
        node.value *= (node.number / (node.number - 1))
        return node.value / node.number

    qs = []
    for i, child in enumerate(node.links):
        if child is None:
            qs.append(U_COEF * (0.3 + node.probs[i]) * (wave_n ** 0.5))
        else:
            qs.append(child.value / child.number + U_COEF * (0.3 +
                                                             node.probs[i]) * (wave_n ** 0.5) / (child.number + 1))

    action = int(np.argmax(qs))
    if node.links[action] is None:
        obs, reward, done, info = trainer.step(action)
        # print(obs, reward, done, info)
        probs, value = agent(obs)
        if done:
            value = reward
        if done and reward is None:
            value = torch.tensor(-1.0)
        node.links[action] = Node(value, probs, done)
        return value
    obs, reward, done, info = trainer.step(action)
    value = wave(node.links[action], agent, trainer, wave_n)
    node.value += value
    node.number += 1

    return value


class SimpleNet(torch.nn.Module):
    def __init__(self, input, actions):
        super(SimpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(input, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.action_layer = torch.nn.Linear(64, actions)
        self.value_layer = torch.nn.Linear(64, 1)
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        action = self.action_layer(x)
        value = self.value_layer(x)
        value = self.tanh(value)
        return action, value


class ComplexNet(torch.nn.Module):
    def __init__(self, input, actions):
        super(SimpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(input, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 64)
        self.layer4 = torch.nn.Linear(64, 64)
        self.action_layer = torch.nn.Linear(64, actions)
        self.value_layer = torch.nn.Linear(64, 1)
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()
        self.activation3 = torch.nn.ReLU()
        self.activation4 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        actions = self.action_layer(x)
        value = self.value_layer(x)
        value = self.tanh(value)
        return


model = SimpleNet(42, 7)
model.load_state_dict({str(net.state_dict())})

def agent():
    return netAgent(model, incorrect_moves=False)"""

file.write(string)
