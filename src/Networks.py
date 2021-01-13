from torch import nn, optim
from torch.nn import functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self, kwargs):
        super(Discriminator, self).__init__()

        ''' receive the state '''
        self.branch1_fc1 = nn.Linear(kwargs['input_dim'][0], 1024)
        self.branch1_fc2 = nn.Linear(1024, 512)
        self.branch1_fc3 = nn.Linear(512 , 256)

        ''' receive the action '''
        self.branch2_fc1 = nn.Linear(kwargs['input_dim'][1], 1024)
        self.branch2_fc2 = nn.Linear(1024, 512)
        self.branch2_fc3 = nn.Linear(512, 256)

        self.com_fc1 = nn.Linear(512, 256)
        self.com_fc2 = nn.Linear(256, kwargs['output_dim'])

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, state, action):
        x1 = self.leaky_relu(self.branch1_fc1(state))
        x1 = self.leaky_relu(self.branch1_fc2(x1))
        x1 = self.leaky_relu(self.branch1_fc3(x1))

        x2 = self.leaky_relu(self.branch2_fc1(action))
        x2 = self.leaky_relu(self.branch2_fc2(x2))
        x2 = self.leaky_relu(self.branch2_fc3(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.leaky_relu(self.com_fc1(x))
        x = self.com_fc2(x)

        return x

class InverseModel(nn.Module):
    def __init__(self, kwargs):
        super(InverseModel, self).__init__()

        ''' receive the state '''
        self.branch1_fc1 = nn.Linear(kwargs['input_dim'][0], 1024)
        self.branch1_fc2 = nn.Linear(1024, 512)
        self.branch1_fc3 = nn.Linear(512, 256)

        ''' receive the next_state '''
        self.branch2_fc1 = nn.Linear(kwargs['input_dim'][1], 1024)
        self.branch2_fc2 = nn.Linear(1024, 512)
        self.branch2_fc3 = nn.Linear(512, 256)

        self.com_fc1 = nn.Linear(512, 256)
        self.com_fc2 = nn.Linear(256, kwargs['output_dim'])

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, next_state):
        x1 = self.leaky_relu(self.branch1_fc1(state))
        x1 = self.leaky_relu(self.branch1_fc2(x1))
        x1 = self.leaky_relu(self.branch1_fc3(x1))

        x2 = self.leaky_relu(self.branch2_fc1(next_state))
        x2 = self.leaky_relu(self.branch2_fc2(x2))
        x2 = self.leaky_relu(self.branch2_fc3(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.leaky_relu(self.com_fc1(x))
        x = self.softmax(self.com_fc2(x))

        ''' predicted action '''
        return x

class ForwardModel(nn.Module):
    def __init__(self, kwargs):
        super(ForwardModel, self).__init__()

        ''' receive the state '''
        self.branch1_fc1 = nn.Linear(kwargs['input_dim'][0], 1024)
        self.branch1_fc2 = nn.Linear(1024, 512)
        self.branch1_fc3 = nn.Linear(512, 256)

        ''' receive the fake action '''
        self.branch2_fc1 = nn.Linear(kwargs['input_dim'][1], 1024)
        self.branch2_fc2 = nn.Linear(1024, 512)
        self.branch2_fc3 = nn.Linear(512, 256)

        self.com_fc1 = nn.Linear(512, 256)
        self.com_fc2 = nn.Linear(256, kwargs['output_dim'])

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, fake_action):
        x1 = self.leaky_relu(self.branch1_fc1(state))
        x1 = self.leaky_relu(self.branch1_fc2(x1))
        x1 = self.leaky_relu(self.branch1_fc3(x1))

        x2 = self.leaky_relu(self.branch2_fc1(fake_action))
        x2 = self.leaky_relu(self.branch2_fc2(x2))
        x2 = self.leaky_relu(self.branch2_fc3(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.leaky_relu(self.com_fc1(x))
        x = self.softmax(self.com_fc2(x))

        ''' predicted next_state '''
        return x
