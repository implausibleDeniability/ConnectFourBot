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
        print(x.shape)
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
        super(ConvNet, self).__init__()

        self.layer1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        # self.layer2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        # self.layer3 = torch.nn.Conv2d(32, 32, 3, padding = 1)
        self.layer4 = torch.nn.Conv2d(32, width, 3, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d([1, 1])

        self.action_layer = torch.nn.Linear(width, actions)
        self.value_layer = torch.nn.Linear(width, 1)

        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()
        self.activation3 = torch.nn.ReLU()
        self.activation4 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        assert len(x.shape) == 1, "PROBLEM"
        x = x.view([1, 1, 6, 7])

        x = self.layer1(x)
        x = self.activation1(x)
        # x = self.layer2(x)
        # x = self.activation2(x)
        # x = self.layer3(x)
        # x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        x = self.avgpool(x)

        x = x.view(1, -1)
        actions = self.action_layer(x)
        value = self.value_layer(x)
        value = self.tanh(value)
        return actions[0], value[0]