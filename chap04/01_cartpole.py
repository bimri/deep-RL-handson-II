#!/usr/bin/env python3

import gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def iterate_batches(environment, network, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observations = environment.reset()
    max_obs_len = max(map(len, observations))

    while True:
        observation_vectors = []
        for observation in observations:
            if len(observation) < max_obs_len:
                observation = np.concatenate((observation, [0] * (max_obs_len - len(observation)))) if observation else np.zeros(max_obs_len)
            observation_vectors.append(observation)
        observation_vectors = torch.from_numpy(np.array(observation_vectors)).float().unsqueeze(0)
        action_probabilities = network(observation_vectors)
        action_probs = action_probabilities.data.cpu().numpy()[0]
        action_probs = action_probs[: len(observations)]
        probabilities = action_probs.ravel() / action_probs.sum()
        action = np.random.choice(len(action_probs), p=probabilities)
        next_observations, reward, is_done, _ = environment.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=observations, action=action)
        episode_steps.append(step)
        if is_done:
            episode = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(episode)
            episode_reward = 0.0
            episode_steps = []
            next_observations = environment.reset()
            max_obs_len = max(map(len, next_observations))
            if len(batch) == batch_size:
                yield batch
                batch = []
        observations = next_observations


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
