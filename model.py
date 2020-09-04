import torch


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


class ResNet(torch.nn.Module):
    def __init__(self, input, actions, width=64):
        super(ResNet, self).__init__()
        self.layer1 = torch.nn.Linear(input, width)
        self.res1 = self.ResBlock(width)
        self.res2 = self.ResBlock(width)
        self.res3 = self.ResBlock(width)
        self.action_layer = torch.nn.Linear(width, actions)
        self.value_layer = torch.nn.Linear(width, 1)
        self.activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        actions = self.action_layer(x)
        value = self.value_layer(x)
        value = self.tanh(value)
        return actions, value

    class ResBlock(torch.nn.Module):
        def __init__(self, width):
            super(ResNet.ResBlock, self).__init__()
            self.layer = torch.nn.Linear(width, width)
            self.activation = torch.nn.ReLU()
        
        def forward(self, x):
            return self.activation(self.layer(x)) * 0.1 + x


class ConvNet(torch.nn.Module):
    def __init__(self, input, actions, width=64):
        super(ResNet, self).__init__()
        self.layer1 = torch.nn.Linear(input, width)
        self.res1 = self.ResBlock(width)
        self.res2 = self.ResBlock(width)
        self.res3 = self.ResBlock(width)
        self.action_layer = torch.nn.Linear(width, actions)
        self.value_layer = torch.nn.Linear(width, 1)
        self.activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        actions = self.action_layer(x)
        value = self.value_layer(x)
        value = self.tanh(value)
        return actions, value

    class ResBlock(torch.nn.Module):
        def __init__(self, width):
            super(ResNet.ResBlock, self).__init__()
            self.layer = torch.nn.Linear(width, width)
            self.activation = torch.nn.ReLU()

        def forward(self, x):
            return self.activation(self.layer(x)) * 0.1 + x
