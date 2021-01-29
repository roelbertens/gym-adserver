import argparse
import sys
import time

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import gym_adserver

class EpsilonGreedyAgent(object):
    def __init__(self, action_space, seed, epsilon):
        self.name = "epsilon-Greedy Agent"
        self.n_actions = action_space.shape[0]
        self.np_random = RandomState(seed)
        self.epsilon = epsilon

    def act(self, users, reward, done) -> None:

        for user in users:
            ads = user.ads

            # Exploitation: choose the ads with the highest CTR so far
            ad_indices = np.argsort([ad.ctr() for ad in ads])[-self.n_actions:]

            # Exploration: for each ad flip coin to determine to keep or discard
            selected = set([])
            n_random = 0
            unchosen_ads = set(range(len(ads))) - set(ad_indices)
            for ad_index in ad_indices:
                if np.random.uniform() < self.epsilon:
                    unchosen_ads.add(ad_index)
                    n_random += 1
                else:
                    selected.add(ad_index)

            user.act(selected | set(np.random.choice(list(unchosen_ads), n_random, replace=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--basket_size', type=int, default=2)
    parser.add_argument('--nr_users', type=int, default=2)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, basket_size=args.basket_size, nr_users=args.nr_users, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)

    # Setup the agent
    agent = EpsilonGreedyAgent(env.action_space, args.seed, args.epsilon)

    # Simulation loop
    reward = 0
    done = False
    users = env.reset(f"{agent.name}: epsilon {args.epsilon}")

    for i in range(args.impressions):
        # Action/Feedback
        agent.act(users, reward, done)
        users, reward, done, _ = env.step()  #actions are passed via users
        
        # Render the current state
        if i % time_series_frequency == 0:
            env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)
    
    env.close()