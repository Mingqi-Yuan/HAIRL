import gym
import torch
import warnings
warnings.filterwarnings('ignore')

from src.PPO import PPOAgent

ENV_NAME = 'Acrobot-v1'
RANDOM_SEED = 2
EPOCHS = 10000
BATCH_SIZE = 64
BATCHES = 1
LR = 1e-5

env = gym.make(ENV_NAME)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

agent = PPOAgent(
    env=env,
    device=device,
    actor_kwargs={'input_dim':6, 'output_dim':env.action_space.n},
    critic_kwargs={'input_dim':6, 'output_dim':1}
)

print('INFO: Train the expert of the env={}'.format(ENV_NAME))
for epoch in range(EPOCHS):
    episode_reward = 0
    state = env.reset()

    while True:
        action = agent.decide(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, done)
        episode_reward += reward

        if done:
            break

        state = next_state

    print('INFO: Round={}, Episode reward={}'.format(epoch + 1, episode_reward))

    ''' save model '''
    if (epoch + 1) % 500 == 0:
        agent.save(save_dir='./models/{}_epochs{}.pth'.format(ENV_NAME, epoch + 1))