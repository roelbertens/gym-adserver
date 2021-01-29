from gym_adserver.envs.ad import Ad

class User:
    def __init__(self, id, num_ads, time_series_frequency, np_random, click_probabilities = None):
        self.id = str(id)
        self.time_series_frequency = time_series_frequency
        self.np_random = np_random
        self.ads = [Ad(i) for i in range(num_ads)]
        self.impressions = 0
        self.clicks = 0
        self.ctr_time_series = []
        self.click_probabilities = click_probabilities
        if self.click_probabilities is None:
            self.click_probabilities = [self.np_random.uniform() * 0.5 for _ in
                                        range(len(self.ads))]  # TODO: initialize with prior add prob

    def step(self):
        """Update clicks and impressions and determine reward depending on chosen action"""

        reward = 0
        for ad_index in self.selected_ad_indices:
            # Update impressions
            self.impressions += 1
            self.ads[ad_index].impressions += 1

            # Update clicks
            if self.np_random.uniform() <= self.click_probabilities[ad_index]:
                self.clicks += 1
                self.ads[ad_index].clicks += 1
                reward += 1

        # Update the ctr time series (for rendering)
        if self.impressions % self.time_series_frequency == 0:
            ctr = 0.0 if self.impressions == 0 else float(self.clicks / self.impressions)
            self.ctr_time_series.append(ctr)

        return reward

    def act(self, selected_ad_indices):
        """The chosen ads"""
        self.selected_ad_indices = selected_ad_indices

    def __repr__(self):
        # return " ".join([f"({ad.clicks}/{ad.impressions})" for ad in self.ads])
        return f"({self.clicks}/{self.impressions})"
    
    def __str__(self):
        return f"User: {self.id}, nr_adds: {len(self.ads)}"

    def __eq__(self, other) : 
        return self.id == other.id
