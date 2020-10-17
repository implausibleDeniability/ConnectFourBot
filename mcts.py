import numpy as np
from kaggle_environments import make
import torch


ACTIONS_N = 7
U_COEF = 2
MCTS_WAVES = 400
TEMPERATURE = 0.5


class Node:
    def __init__(self, value, probs, done=False):
        self.value = value
        self.probs = probs
        self.number = 1
        self.done = done
        self.links = [None] * ACTIONS_N


def mcts(observation, state, agent, against, root=None):
    if root is None:
        probs, value = agent(observation)
        root = Node(value, probs)
    current_n = root.number

    n_traverses = MCTS_WAVES - current_n
    for i in range(n_traverses):
        current_n += 1
        env = make("connectx", debug = False)
        if (observation['mark'] == 1):
            trainer = env.train([None, against])
        else:
            trainer = env.train([against, None]) 
        env.state = state.copy()
        wave(root, agent, trainer, current_n)

    policy = []
    for child in root.links:
        if child is None:
            policy.append(0)
        else:
            policy.append(child.number)
    policy = np.array(policy)
    policy = policy ** (1 / TEMPERATURE)
    policy = policy / policy.sum()

    return policy, root
    
        
def wave(node, agent, trainer, wave_n):
    if (node.done):
        node.number += 1
        node.value *= (node.number / (node.number - 1))
        return node.value / node.number

    qs = []
    for i, child in enumerate(node.links):
        if child is None:
            qs.append(U_COEF * (0.2 + node.probs[i]) * (wave_n ** 0.5))
        else:
            qs.append(child.value / child.number + U_COEF * (0.2 + node.probs[i]) * (wave_n ** 0.5) / (child.number + 1))
        
    
    action = int(np.argmax(qs))
    if node.links[action] is None:
        obs, reward, done, info = trainer.step(action)
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
