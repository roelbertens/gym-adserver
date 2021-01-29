import gym
from gym import logger, spaces
from gym.utils import seeding

import numpy as np
from numpy.random.mtrand import RandomState

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['toolbar'] = 'None'

from gym_adserver.envs.user import User
from gym_adserver.envs.ad import Ad

class AdServerEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, num_ads, basket_size, nr_users, time_series_frequency, reward_policy=None):
        self.time_series_frequency = time_series_frequency        
        self.num_ads = num_ads
        self.basket_size = basket_size
        self.nr_users = nr_users
        self.reward_policy = reward_policy

        # Environment OpenAI metadata
        # self.reward_range = (0, 1)
        # self.action_space = spaces.MultiDiscrete([num_ads]*basket_size)
        self.action_space = spaces.Box(low=0, high=num_ads-1, shape=(nr_users, basket_size), dtype=np.int)  # TODO space shouldnt include same ads for a user
        # self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2, num_ads), dtype=np.float) # clicks and impressions, for each ad

    def seed(self, seed=None): # pragma: no cover
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self):
        """Depending on the probabilities update the clicks/impressions for each user

        Note: the action is stored for each user.

        Returns:
            ?
        """
        reward = 0
        for user in self.users:
            reward += user.step()

        return self.users, reward, False, {}

    def reset(self, scenario_name):
        self.scenario_name = scenario_name
        self.users = [
            User(id=i, num_ads=self.num_ads, time_series_frequency=self.time_series_frequency, np_random=self.np_random)
            for i in range(self.nr_users)]
        return self.users

    def render(self, mode='human', freeze=False, output_file=None): # pragma: no cover
        if mode != 'human':
            raise NotImplementedError

        max_users = 2
        fig = plt.figure(num=self.scenario_name, figsize=(8, 4*max_users))
        grid_size = (5*max_users, 2)
        row_start = 0

        for user_id, user in enumerate(self.users):
            if user_id >= max_users:
                break

            ads = user.ads
            impressions = user.impressions
            clicks = user.clicks
            ctr_time_series = user.ctr_time_series
            click_probabilities = user.click_probabilities

            ctr = 0.0 if impressions == 0 else float(clicks / impressions)

            logger.info('Impressions: {}, CTR: {}, Ads: {}'.format(impressions, ctr, ads))


            # Plot CTR time series
            plt.subplot2grid(grid_size, (row_start, 0), rowspan=2, colspan=2)
            x = [i for i, _ in enumerate(ctr_time_series)]
            y = ctr_time_series
            axes = plt.gca()
            axes.set_ylim([0,None])
            plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
            plt.ylabel("CTR")
            plt.xlabel("Impressions")
            plt.plot(x, y, marker='o')
            for x,y in zip(x,y):
                plt.annotate("{:.2f}".format(y), (x,y), textcoords="offset points", xytext=(0,10), ha='center')

            # Plot impressions
            plt.subplot2grid(grid_size, (row_start+2, 0), rowspan=3, colspan=1)
            x = [ad.id for ad in ads]
            impressions = [ad.impressions for ad in ads]
            x_pos = [i for i, _ in enumerate(x)]
            plt.barh(x_pos, impressions)
            plt.ylabel("Ads")
            plt.xlabel("Impressions")
            plt.yticks(x_pos, x)

            # Plot CTRs and probabilities
            plt.subplot2grid(grid_size, (row_start+2, 1), rowspan=3, colspan=1)
            x = [ad.id for ad in ads]
            y = [ad.ctr() for ad in ads]
            y_2 = click_probabilities
            x_pos = [i for i, _ in enumerate(x)]
            x_pos_2 = [i + 0.4 for i, _ in enumerate(x)]
            plt.ylabel("Ads")
            plt.xlabel("")
            plt.yticks(x_pos, x)
            plt.barh(x_pos, y, 0.4, label='Actual CTR')
            plt.barh(x_pos_2, y_2, 0.4, label='Probability')
            plt.legend(loc='upper right')

            row_start += 5

        plt.tight_layout()

        if output_file is not None:
            fig.savefig(output_file)

        if freeze:
            # Keep the plot window open
            # https://stackoverflow.com/questions/13975756/keep-a-figure-on-hold-after-running-a-script
            if matplotlib.is_interactive():
                plt.ioff()
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.001)

    def close(self):
        plt.close()