import torch
import gym
import warnings
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter

from src.HAIRL import *
from src.PPO import *

ENV_NAME = 'Acrobot-v1'
EXPERT_AGENT = torch.load('./models/gym/{}_epochs500.pth'.format(ENV_NAME))
RANDOM_SEED = 2
EPOCHS = 500
BATCH_SIZE = 64
BATCHES = 1
LR = 1e-5
SHUFFLE = True

env = gym.make(ENV_NAME)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./log/hairl')

''' Network kwargs '''
ACTOR_KWARGS = {'input_dim':6, 'output_dim':env.action_space.n}
CRITC_KWARGS = {'input_dim':6, 'output_dim':1}
DIS_KWARGS = {'input_dim':[6, env.action_space.n], 'output_dim':1}
INVERSE_KWARGS = {'input_dim':[6,6], 'output_dim':env.action_space.n}
FORWARD_KWARGS = {'input_dim':[6, env.action_space.n], 'output_dim':6}

hairl = HAIRL(
    env=env,
    device=device,
    writer=writer,
    actor_kwargs=ACTOR_KWARGS,
    critic_kwargs=CRITC_KWARGS,
    dis_kwargs=DIS_KWARGS,
    inverse_kwargs=INVERSE_KWARGS,
    forward_kwargs=FORWARD_KWARGS,
    expert_agent=EXPERT_AGENT,
    lr=LR,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE
)

for epoch in range(EPOCHS):
    hairl.train(epoch)