from kaggle_environments import make

def evaluate(agent, against, n_games):
    result = 0
    for i in range(n_games // 2):
        env = make('connectx', debug=True)
        laststate = env.run([agent, against])[-1]
        gamereward = laststate[0]['reward']
        result += (gamereward + 1) / 2

        laststate = env.run([against, agent])[-1]
        gamereward = laststate[1]['reward']
        result += (gamereward + 1) / 2
    return result / n_games
