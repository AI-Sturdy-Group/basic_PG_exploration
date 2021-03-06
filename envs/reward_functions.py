class LinearReward(object):

    def __init__(self, max_reward: float = 1., min_reward: float = 0., min_action: float = -1.,
                 max_action: float = 1., target_action: float = 0.5):

        assert min_reward < max_reward
        assert min_action < max_action
        assert min_action <= target_action <= max_action

        self.max_reward = max_reward
        self.min_reward = min_reward
        self.min_action = min_action
        self.max_action = max_action
        self.target_action = target_action

    def calculate_reward(self, action: float):

        if action == self.target_action:
            return self.max_reward
        elif action < self.target_action:
            reward = action - self.min_action
            reward = reward / (self.target_action - self.min_action)
            reward = reward * (self.max_reward - self.min_reward)
            reward = reward + self.min_reward
            return reward
        elif action > self.target_action:
            reward = action - self.target_action
            reward = reward / (self.max_action - self.target_action)
            reward = reward * (self.min_reward - self.max_reward)
            reward = reward + self.max_reward
            return reward
