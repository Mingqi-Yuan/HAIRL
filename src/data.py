import torch
import numpy as np
from torch.utils import data

def build_loader(
        trajs_df,
        trajs_length,
        batch_size,
        shuffle,
        drop_last=True,
        num_workers=1):

    states = np.stack(trajs_df['state']).astype('float32')
    actions = np.stack(trajs_df['action'])
    rewards = np.stack(trajs_df['reward'])

    dataset = data.TensorDataset(
        torch.from_numpy(states[:trajs_length - 1]),
        torch.from_numpy(actions[:trajs_length - 1]),
        torch.from_numpy(rewards[:trajs_length - 1]),
        torch.from_numpy(states[1:trajs_length])
    )

    data_loader = data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return data_loader

