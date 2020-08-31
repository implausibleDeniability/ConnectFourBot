from agents import netAgent, processObservation
from mcts import mcts
from model import SimpleNet
from torch import nn
from kaggle_environments import make
import numpy as np
import torch
from collections import OrderedDict

netstr = OrderedDict([('layer1.weight', tensor([[ 0.1006, -0.0153, -0.0361,  ..., -0.0060,  0.0872, -0.0631],         [-0.1536,  0.0686, -0.0522,  ..., -0.0065,  0.0018,  0.0403],         [-0.1534, -0.0249, -0.1457,  ..., -0.0181, -0.0755,  0.1505],         ...,         [-0.0979, -0.0030, -0.0299,  ...,  0.1264, -0.0991, -0.0334],         [-0.0667,  0.1425,  0.0931,  ..., -0.1503,  0.0464,  0.1441],         [-0.1432,  0.0909,  0.0262,  ...,  0.0136,  0.0025,  0.0251]])), ('layer1.bias', tensor([ 0.0077,  0.0860, -0.0526, -0.1231,  0.1409, -0.0758,  0.0836,  0.1245,          0.0981, -0.0303, -0.0019,  0.0153,  0.1460,  0.0234,  0.0837, -0.0682,         -0.0502, -0.0722,  0.0814, -0.0674, -0.1368, -0.0843,  0.1282, -0.1255,         -0.0286,  0.1400, -0.0773,  0.1245,  0.1291, -0.0302,  0.0005,  0.0208,         -0.1288,  0.0718,  0.0594, -0.1455, -0.1146, -0.1444, -0.0800,  0.0580,         -0.0704,  0.0942,  0.0426, -0.1208,  0.0205, -0.0390, -0.0505,  0.1301,          0.1460, -0.0477,  0.1186,  0.0341,  0.0850,  0.0655, -0.1382, -0.0769,          0.1198,  0.0761,  0.0003,  0.0188, -0.0045,  0.0836,  0.1496, -0.0801])), ('layer2.weight', tensor([[ 0.0273, -0.0796,  0.0274,  ..., -0.0039,  0.1154, -0.0642],         [ 0.0240,  0.0081, -0.0288,  ...,  0.0451,  0.0609,  0.0073],         [ 0.0787, -0.0966, -0.0325,  ..., -0.0234, -0.0909,  0.0018],         ...,         [-0.0838, -0.0908,  0.0650,  ...,  0.0764, -0.0459,  0.0897],         [ 0.0831, -0.0584, -0.0210,  ..., -0.1167,  0.0595,  0.0378],         [-0.0144, -0.0899, -0.1096,  ...,  0.0192,  0.1184,  0.1203]])), ('layer2.bias', tensor([-0.0173,  0.1227, -0.0124,  0.0837, -0.1079,  0.0363,  0.0442,  0.0334,          0.1236,  0.0081,  0.0017,  0.0732,  0.0191,  0.1001,  0.0498, -0.0924,          0.0585,  0.1015,  0.1213,  0.0848,  0.0968, -0.0946, -0.0275, -0.0212,          0.1002, -0.0998,  0.0071, -0.0449, -0.0257,  0.1176,  0.0978,  0.0424,          0.0701, -0.0695,  0.0720, -0.0191, -0.0912, -0.0818, -0.0322, -0.0112,          0.0125,  0.1212, -0.0094, -0.0393, -0.0569, -0.0692,  0.0544,  0.0900,          0.0545,  0.0150,  0.0258, -0.0526,  0.0767, -0.1001,  0.0525,  0.0733,          0.0837,  0.0901,  0.1229,  0.0468, -0.1246, -0.1087, -0.0587,  0.0424])), ('action_layer.weight', tensor([[-0.0153, -0.0108, -0.0640, -0.0042, -0.0621, -0.0919,  0.0870,  0.0276,           0.1082,  0.0142, -0.0722, -0.0842, -0.0057,  0.0992, -0.0371,  0.0853,           0.1042,  0.0430,  0.0221, -0.0398,  0.0016,  0.1048, -0.0194, -0.0362,          -0.0610, -0.0831,  0.0065,  0.0268, -0.0949, -0.1202,  0.1194,  0.0992,          -0.1022, -0.0895,  0.0716, -0.0740, -0.0583, -0.0683,  0.0273,  0.0501,           0.1140, -0.0380, -0.0108, -0.1183, -0.0191, -0.0845, -0.0237,  0.0235,          -0.0715, -0.0903,  0.0613, -0.0493,  0.0466, -0.0735, -0.1096, -0.0313,          -0.1130, -0.0362, -0.0169, -0.0288,  0.0872, -0.0005, -0.0852,  0.0540],         [ 0.0514,  0.0833,  0.1024, -0.1237, -0.0775, -0.1141, -0.0789,  0.0394,          -0.0213, -0.1250, -0.0672, -0.0010, -0.0730,  0.0600, -0.0154,  0.1249,           0.1161,  0.0833, -0.0491, -0.1193, -0.0813, -0.0957, -0.0060, -0.0423,           0.0541, -0.0928,  0.0606, -0.0827,  0.1192, -0.0239,  0.0499,  0.0319,          -0.0269,  0.0178, -0.0490, -0.0993, -0.0622, -0.0696, -0.0994,  0.0245,          -0.0561,  0.0812,  0.0116,  0.0859, -0.1008,  0.0320, -0.1226,  0.0611,           0.0602, -0.0594, -0.1053,  0.1127,  0.0571,  0.0340,  0.0456, -0.0469,           0.0281,  0.0315, -0.0675,  0.0073, -0.0718, -0.0395,  0.0119,  0.0739],         [ 0.0230,  0.0854, -0.0578,  0.0160, -0.0793, -0.1231,  0.0972, -0.0193,           0.1029,  0.0972, -0.0349,  0.0039,  0.0262, -0.0689,  0.0421,  0.0278,           0.1138, -0.0321, -0.0595, -0.0576,  0.1175,  0.0565, -0.1235,  0.1029,           0.0561, -0.1236,  0.0131, -0.0908,  0.0409,  0.0612, -0.0241, -0.1133,           0.0286, -0.0774, -0.0588, -0.0697, -0.0048, -0.0268,  0.0118, -0.0883,           0.0373, -0.0846, -0.1211, -0.0160, -0.0618,  0.1082, -0.0470,  0.0943,           0.1242,  0.0159,  0.0319, -0.1109, -0.0765, -0.1061, -0.0256, -0.0585,          -0.0094, -0.0215, -0.0472, -0.0905, -0.1204, -0.0173, -0.0152,  0.0803],         [-0.0099,  0.1201, -0.1184, -0.1005, -0.1209,  0.1046,  0.0160,  0.0153,          -0.1242,  0.1149, -0.0709, -0.1202,  0.1215,  0.0222,  0.0342, -0.0821,          -0.0778,  0.1013,  0.0935, -0.0570,  0.0902,  0.1213,  0.0545,  0.0208,          -0.0667, -0.0023, -0.0822,  0.0829, -0.1167,  0.0557, -0.0224,  0.0502,           0.0015,  0.0612, -0.0337, -0.0613,  0.1032, -0.0136,  0.0736,  0.0886,           0.0045, -0.1084, -0.1108, -0.0477,  0.0187, -0.1086, -0.1069,  0.0281,           0.0789,  0.0395, -0.0754, -0.0505, -0.0849,  0.0857, -0.1203,  0.1105,          -0.0880, -0.0953,  0.0394,  0.0983,  0.1183,  0.0089,  0.0145,  0.0748],         [-0.0799, -0.0475,  0.1039,  0.0489,  0.1070, -0.0118,  0.0448,  0.1114,           0.0569,  0.1218,  0.0708,  0.0867, -0.0358,  0.0789,  0.1235,  0.0092,           0.0485,  0.0562,  0.0098,  0.0045, -0.0063, -0.0430,  0.0880, -0.0872,           0.0030,  0.1106, -0.0489, -0.0110,  0.0391, -0.0887,  0.0582,  0.0734,           0.1182,  0.0566, -0.0865, -0.0311,  0.0155, -0.1122, -0.1060,  0.0947,           0.0779,  0.0959, -0.0628, -0.0113,  0.1198, -0.0690, -0.0145, -0.0985,          -0.0579, -0.0165, -0.0463,  0.1086, -0.0461, -0.0245, -0.1063,  0.0632,          -0.0466, -0.0472,  0.0541,  0.0724, -0.0673, -0.0391,  0.1126,  0.1139],         [ 0.0151, -0.0873,  0.0574, -0.0177, -0.1074, -0.0909, -0.0800,  0.0959,           0.0066,  0.0548, -0.0949, -0.0776, -0.0789,  0.0508,  0.0301, -0.0447,           0.1180, -0.0479, -0.0992, -0.1109,  0.0414, -0.0899,  0.0474,  0.1209,           0.0378,  0.0510,  0.0312, -0.1082, -0.0003,  0.0016, -0.1107, -0.1163,           0.0192,  0.0332, -0.0300,  0.0724, -0.1163,  0.0755, -0.1221, -0.0591,           0.0096, -0.0016, -0.0189, -0.1051, -0.0813, -0.0098, -0.0292, -0.1087,           0.0550, -0.0138,  0.0693, -0.0926, -0.0748,  0.0697, -0.0636,  0.1072,           0.0239,  0.0703, -0.1238, -0.0039, -0.0440, -0.0026, -0.1150,  0.0627],         [ 0.0745, -0.1220,  0.0169, -0.0964, -0.0466,  0.1237,  0.0618, -0.0189,           0.0619, -0.0107, -0.0155, -0.0595, -0.0709, -0.0854, -0.0598,  0.0012,           0.0687, -0.0462, -0.0311, -0.0332, -0.1074,  0.1076, -0.0517, -0.1200,          -0.1178, -0.1112, -0.1221, -0.0270, -0.0299, -0.0786, -0.1012,  0.0225,           0.0136,  0.0197, -0.0631,  0.0086,  0.0952,  0.0621, -0.0243, -0.0660,          -0.0412,  0.0864, -0.0976,  0.0400, -0.0318,  0.1100, -0.1144, -0.0925,          -0.0563, -0.0278, -0.0196,  0.0079, -0.0109,  0.0222,  0.0044,  0.1035,           0.0359,  0.0472,  0.0206, -0.0793,  0.0958, -0.0955,  0.1227,  0.0821]])), ('action_layer.bias', tensor([ 0.0505,  0.0590,  0.1072, -0.0872, -0.0970,  0.0503, -0.0449])), ('value_layer.weight', tensor([[ 0.1021,  0.1247,  0.0836, -0.0993,  0.0399,  0.0038,  0.0039,  0.0979,           0.0005,  0.0845,  0.1028, -0.0264, -0.0551, -0.0338,  0.0145,  0.0276,          -0.0723, -0.0467, -0.0868, -0.0751,  0.0519,  0.0503, -0.0348,  0.0259,          -0.0608, -0.1086,  0.1099,  0.0781,  0.1034,  0.0490, -0.0175,  0.1032,          -0.0629, -0.0581,  0.0992, -0.0416, -0.0420, -0.0107,  0.0191, -0.0598,           0.0735, -0.0905,  0.0659,  0.0434,  0.0880,  0.0784,  0.0035, -0.0604,           0.0934,  0.0387,  0.0564,  0.0054,  0.0188,  0.0154, -0.0369,  0.0396,           0.0405,  0.0272,  0.0212,  0.1053, -0.1191,  0.0517, -0.0919,  0.0387]])), ('value_layer.bias', tensor([0.1097]))])

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
model.load_state_dict(netstr)

def agent():
    return netAgent(model, incorrect_moves=False)