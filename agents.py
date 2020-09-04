import numpy as np
import torch


def processObservation(observation):
    board = np.array(observation['board'])
    if (observation['mark'] == 2):
        ones = np.argwhere(board == 1)
        twos = np.argwhere(board == 2)
        board[ones] = 2
        board[twos] = 1
    return board


def netAgent(network, return_probs=False,
        incorrect_moves=True, best_move=True):
    if return_probs:
        def agent(observation, configuration=None):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            with torch.no_grad():
                output = network(input)
            probs = torch.softmax(output[0], 0)
            return probs, output[1]

    elif incorrect_moves and best_move:
        def agent(observation, configuration=None):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            with torch.no_grad():
                output = network(input)
            probs = torch.softmax(output[0], 0)
            return torch.argmax(probs).item()
    
    elif incorrect_moves and not best_move:
        def agent(observation, configuration=None):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            with torch.no_grad():
                output = network(input)
            probs = torch.softmax(output[0], 0)
            return int(np.random.choice(range(7), p=probs.numpy()))
            
    elif not incorrect_moves and best_move:
        def agent(observation, configuration=None):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            with torch.no_grad():
                output = network(input)
            probs = torch.softmax(output[0], 0)
            action = torch.argmax(probs).item()
            while board[action] != 0:
                probs[action] = 0
                action = torch.argmax(probs).item()
            return action

    elif not incorrect_moves and not best_move:
        def agent(observation, configuration=None, incorrect_moves=incorrect_moves, best_move=best_move):
            board = processObservation(observation)
            input = torch.tensor(board, dtype=torch.float32)
            with torch.no_grad():
                output = network(input)
            probs = torch.softmax(output[0], 0)
            action = np.random.choice(range(7), p=probs.numpy())
            while board[action] != 0:
                probs += probs[action] / 6
                probs[action] = 0
                action = np.random.choice(range(7), p=probs.numpy())
            return int(action)

    return agent
