from kaggle_environments import make


def act(observation, configuration):
    board = observation.board
    columns = configuration.columns
    return [c for c in range(columns) if board[c] == 0][0]
    
env = make("connectx")
state = env.step([0, 1, 2])
print(state)
