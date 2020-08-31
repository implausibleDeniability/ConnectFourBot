from model import SimpleNet
import torch

net = SimpleNet(42, 7)
net.load_state_dict(torch.load('parameters_simple.pth'))
file = open("string.txt", 'w')
string = str(net.state_dict())
string = string.replace('\n', ' ', 1000000)
print(string)
file.write(string)
