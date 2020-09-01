from submission2 import agent
from kaggle_environments import make

env = make("connectx", debug=True)
print(env.play(agent, agent))