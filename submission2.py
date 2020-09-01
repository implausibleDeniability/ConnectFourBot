
from agents import netAgent, processObservation
from mcts import mcts
from model import SimpleNet
from torch import nn
from kaggle_environments import make
import numpy as np
import torch
from collections import OrderedDict
from torch import tensor 

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
model.load_state_dict(OrderedDict([('layer1.weight', tensor([[ 0.0347,  0.0936, -0.1214,  ...,  0.0353, -0.1334,  0.1432],
        [-0.0726, -0.0994,  0.1479,  ...,  0.1181, -0.1305,  0.1488],
        [-0.0725, -0.0575,  0.1177,  ..., -0.0263, -0.0365, -0.0836],
        ...,
        [-0.0674, -0.1255,  0.0631,  ..., -0.1018, -0.0748,  0.0024],
        [-0.0294,  0.1366, -0.0605,  ..., -0.0047, -0.0095, -0.1053],
        [-0.0748,  0.0992, -0.0072,  ...,  0.0264, -0.1184,  0.0303]])), ('layer1.bias', tensor([ 0.0363, -0.0034,  0.0823,  0.1306,  0.1190,  0.1733, -0.0227,  0.0090,
        -0.0673, -0.1342,  0.1018, -0.0677,  0.0031, -0.1153, -0.0966,  0.1186,
         0.1599,  0.0480, -0.0394, -0.0195, -0.0806, -0.0405,  0.0645, -0.1216,
        -0.0606, -0.1727, -0.0343, -0.1492, -0.0449, -0.1448,  0.1610,  0.1115,
         0.1238, -0.0299, -0.0048, -0.1032, -0.0851,  0.0465,  0.1195,  0.0015,
         0.0145, -0.1154, -0.1544, -0.0845, -0.0461,  0.0999,  0.0194, -0.0787,
         0.1077,  0.0149, -0.0928,  0.0587,  0.1101, -0.1113,  0.0921,  0.1333,
         0.0442, -0.0229, -0.1050,  0.0955,  0.0286,  0.0440,  0.0476, -0.1310])), ('layer2.weight', tensor([[ 1.9107e-02,  7.8225e-02,  8.1197e-02,  ..., -1.1939e-01,
          3.0674e-02, -3.8244e-03],
        [ 3.6271e-02, -6.4494e-02,  4.1489e-02,  ...,  1.0411e-01,
          1.1161e-01,  8.1869e-02],
        [-8.8051e-02,  4.4756e-02,  4.9698e-02,  ...,  1.2412e-01,
          1.2210e-01,  2.8085e-02],
        ...,
        [-7.3225e-02,  3.5936e-02, -4.3703e-02,  ..., -4.7991e-02,
          6.9162e-02, -6.9568e-02],
        [ 2.6965e-02, -5.5224e-02,  1.2351e-01,  ..., -1.1152e-04,
          3.8157e-02,  9.0789e-02],
        [ 5.3895e-02,  2.1894e-02, -2.3118e-03,  ...,  6.1012e-02,
          5.6208e-03, -1.2568e-01]])), ('layer2.bias', tensor([ 0.0861, -0.0547,  0.1521,  0.0163, -0.0982,  0.0212,  0.0665, -0.0733,
        -0.0671, -0.0631,  0.1697, -0.0020, -0.1087,  0.0469,  0.0066,  0.0719,
         0.1112, -0.0969, -0.1196,  0.0526,  0.1270,  0.1544, -0.0414, -0.0256,
         0.1717, -0.0192,  0.0621, -0.0647, -0.0125,  0.0906, -0.0761, -0.0311,
         0.0680, -0.0893,  0.0649, -0.0421,  0.0504, -0.0224, -0.0674, -0.1221,
        -0.1649,  0.0421,  0.0866, -0.1000, -0.1051, -0.0816, -0.0090, -0.0810,
        -0.1338,  0.0285, -0.0906,  0.1553,  0.1488,  0.1696,  0.1582, -0.1002,
         0.0268, -0.0868, -0.1175,  0.2125, -0.0979, -0.0807,  0.0718, -0.0720])), ('action_layer.weight', tensor([[ 0.0706,  0.0007, -0.0292,  0.0342,  0.0946,  0.0668, -0.0212, -0.0847,
          0.0715,  0.1385,  0.0882, -0.0621, -0.0943,  0.1020, -0.0804, -0.1310,
         -0.0300,  0.1038, -0.1218,  0.1151, -0.1445,  0.1656, -0.1054, -0.0991,
          0.0318, -0.0330,  0.0344, -0.1132,  0.0741, -0.0449, -0.0461, -0.1045,
          0.0410,  0.0061, -0.0173, -0.0080,  0.0244,  0.0012, -0.0740,  0.0754,
         -0.1110, -0.0271, -0.0873,  0.0845,  0.0241, -0.0715,  0.1406, -0.0473,
         -0.1245,  0.0255, -0.0671,  0.1154,  0.1668, -0.1843,  0.0907,  0.0398,
          0.1725,  0.1206,  0.0838,  0.0824, -0.0728,  0.1186,  0.0826,  0.1440],
        [-0.0056,  0.0040,  0.1116, -0.0284,  0.0347,  0.0142, -0.1069,  0.1066,
         -0.0240,  0.0754,  0.0843,  0.0470,  0.0912, -0.0306, -0.0309, -0.0040,
          0.0354,  0.0731,  0.0553, -0.0651,  0.0524, -0.1261,  0.0299,  0.0893,
         -0.1163, -0.0671,  0.0957, -0.0847,  0.0988, -0.0664, -0.1147,  0.0074,
          0.0929,  0.0401, -0.1070,  0.1231, -0.1153,  0.0779,  0.0867, -0.0880,
          0.1039, -0.0413,  0.0536, -0.0136, -0.0062, -0.0440,  0.0825, -0.0362,
          0.1167, -0.1163,  0.1064, -0.1208,  0.0616, -0.0533,  0.0153, -0.0967,
         -0.1350, -0.0762,  0.1042, -0.1072,  0.0807, -0.0760,  0.0008, -0.0838],
        [ 0.0727, -0.0509, -0.0114, -0.0125,  0.1084, -0.0051, -0.0411,  0.1381,
         -0.0430,  0.0796, -0.1040, -0.0416, -0.0777, -0.0229, -0.0114, -0.0494,
         -0.0176, -0.0277,  0.0152,  0.1042,  0.1175, -0.0387,  0.0896, -0.1011,
         -0.1057,  0.1167,  0.0665,  0.0723,  0.1073,  0.0182, -0.0195,  0.0703,
         -0.1200,  0.1014,  0.0883, -0.0730, -0.0067,  0.0203,  0.0406,  0.1012,
         -0.0751, -0.0202, -0.0925,  0.1273,  0.1132,  0.0391,  0.0881, -0.1099,
         -0.0544,  0.0087,  0.0045, -0.0566,  0.0432,  0.0827, -0.1318, -0.0966,
          0.0785,  0.0036, -0.0420,  0.0855, -0.1134, -0.0524,  0.1421,  0.0003],
        [ 0.0846, -0.0502,  0.0552,  0.0891, -0.0420, -0.0793, -0.0614, -0.0792,
          0.0422, -0.1270,  0.0417,  0.0184, -0.0845, -0.1050, -0.0085, -0.0899,
          0.0673, -0.1004, -0.1126, -0.0788, -0.0535, -0.0758,  0.0253, -0.0156,
          0.0387, -0.0622,  0.0889,  0.0980,  0.0371, -0.0147,  0.1228,  0.0278,
         -0.0104,  0.0888,  0.1027, -0.1044, -0.1222, -0.0814, -0.0493, -0.1036,
         -0.0474,  0.0483, -0.0463,  0.1085, -0.0158, -0.0281,  0.0485, -0.0603,
         -0.0455,  0.0256,  0.0959, -0.0179, -0.1310,  0.0476, -0.1095,  0.0971,
          0.0382, -0.0818, -0.0123, -0.0981, -0.0129,  0.0365,  0.0826,  0.0058],
        [ 0.0586, -0.0908, -0.0829, -0.0927,  0.0912, -0.0389, -0.0600, -0.1002,
          0.0584, -0.1245, -0.0043,  0.0514, -0.1038,  0.0158,  0.1109, -0.0343,
          0.0021, -0.0197, -0.1293,  0.1160,  0.0428, -0.0204, -0.0078,  0.0631,
         -0.1103,  0.0921,  0.0263,  0.0485,  0.0808, -0.0139,  0.0437, -0.0041,
          0.0878, -0.1235, -0.0054, -0.0633,  0.1209, -0.0838, -0.0003,  0.1221,
          0.0682,  0.0570, -0.0150, -0.0103, -0.1007,  0.0965, -0.0547,  0.0355,
         -0.0664, -0.0209, -0.0327,  0.0829, -0.1001,  0.0246, -0.0305,  0.1214,
          0.0839,  0.0078, -0.0436,  0.0378,  0.0173, -0.0311,  0.0525,  0.0458],
        [ 0.0900, -0.0530, -0.0896, -0.0932, -0.0548,  0.0046,  0.1040,  0.0980,
         -0.1044,  0.0424,  0.1200,  0.0782, -0.0710,  0.1108, -0.0114,  0.0013,
          0.0457,  0.0121, -0.0444, -0.0767, -0.0057, -0.0897, -0.0995,  0.0509,
         -0.0101,  0.1483,  0.0195,  0.1055,  0.0477,  0.0329,  0.0523,  0.1110,
         -0.0801,  0.0288,  0.0880, -0.1058,  0.1058, -0.0498, -0.1068,  0.0648,
          0.0737, -0.0811,  0.1132,  0.0846, -0.1073,  0.0514, -0.0095,  0.1135,
          0.1038,  0.0786,  0.0132, -0.0124, -0.0580,  0.0934, -0.1081, -0.0549,
          0.0446,  0.0400,  0.0030, -0.0351, -0.0383,  0.0103,  0.1473, -0.0196],
        [-0.0358,  0.0579, -0.0989, -0.0862, -0.0369, -0.0338,  0.0743, -0.0805,
          0.0169, -0.0427,  0.0772,  0.0162,  0.0965,  0.0140,  0.0604,  0.0888,
         -0.0502,  0.0619,  0.0397,  0.0558,  0.0477, -0.0579, -0.1180,  0.0769,
         -0.0353,  0.0759, -0.0806,  0.0730,  0.1049, -0.0003, -0.0088,  0.0236,
         -0.0780,  0.0789, -0.0521,  0.0235, -0.1106,  0.0702, -0.1245, -0.0282,
         -0.1082, -0.0015, -0.1014, -0.0036, -0.0350, -0.0895, -0.1067, -0.0324,
         -0.1127, -0.0089,  0.1120,  0.0138, -0.1028,  0.0980, -0.0731,  0.0035,
          0.0589,  0.0525,  0.1075, -0.1426, -0.0881, -0.0793, -0.0152, -0.0603]])), ('action_layer.bias', tensor([ 0.0486, -0.0934, -0.0954, -0.1167, -0.1196,  0.0110, -0.1524])), ('value_layer.weight', tensor([[ 0.1713,  0.1034,  0.3641, -0.0547,  0.0711,  0.0060,  0.3378,  0.0690,
          0.2863,  0.0072,  0.1995,  0.1533,  0.0692,  0.0399, -0.0454,  0.1153,
          0.3931,  0.0581, -0.0079,  0.1748,  0.1768,  0.3352,  0.0416, -0.1101,
          0.1130,  0.1534,  0.0643,  0.1671, -0.0416,  0.4609, -0.0103,  0.0626,
          0.1503, -0.0320,  0.1689,  0.1090,  0.2027, -0.0822,  0.0491, -0.0672,
         -0.0651, -0.0169,  0.1564, -0.0716, -0.0496,  0.0759,  0.3559, -0.0441,
         -0.0473,  0.0682, -0.0123,  0.2828,  0.3246,  0.2247,  0.3276, -0.0860,
          0.1786, -0.0921, -0.0843,  0.2533,  0.0247, -0.1011,  0.1620,  0.0762]])), ('value_layer.bias', tensor([1.5132]))]))


correctagent = netAgent(model, incorrect_moves=False)
def agent(observation, configuration):
    return correctagent(observation, configuration)
