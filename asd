import numpy as np
import pandas as pd
import math
from gym import spaces
import gym
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


class StockMarketEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    # (sort, long, idle) or (sort, long)

    num_of_action = 3
    date = 'Date'
    open = 'Open'
    close = 'Close'
    high = 'High'
    low = 'Low'
    new_stock = 'new_stock'

    # FEATURES
    c_o = 'c_o'
    c_l_h = 'c_l_h'
    o_c_1 = 'o_c_1'
    aggr_mean = 'aggr_mean'
    aggr_std = 'aggr_std'

    # state
    HOLD = 'Hold'
    NO_HOLD = 'Not hold'

    def __init__(self):
        super(StockMarketEnv, self).__init__()
        self.action_space = spaces.Discrete(self.num_of_action)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 6), dtype=float)
        #self.data_frame = pd.read_csv("traning_data\\traning_features.csv")
        self.balance = 10000
        self.hold = False
        self.buy_price = 0
        self.iter = 0
        self.data = pd.read_csv("c:\\tmp\\roxenv_sh\\traning_data\\traning_features.csv")
        self.state = self.NO_HOLD
        self.cum_rewarsd = 0
        self.time_between_sell = 0
        self.sellnum = 0

    def calc_reward(self):
        reward = 0
        if self.state is self.HOLD:
            if math.isclose(self.buy_price, 0):
                print("Fuck price")
                print(self.iter)
            else:
                reward = (self.data.Open[self.iter] - self.buy_price) / self.buy_price
        return reward

    def step(self, action):
        reward = 0
        done = False
        self.iter = self.iter + 1
        if self.iter > 2340390:  # 2340390
            print("Reward:" + str(self.cum_rewarsd))
            print("DOOONNNNNEE")
            done = True
        if self.data.new_stock[self.iter] == 0:
            print("New stock")
            print("Reward:" + str(self.cum_rewarsd))
            self.cum_rewarsd = 0
            self.sellnum =0
            action = 2

        # if action == 0:

        if action == 1:
            if self.state is self.NO_HOLD:
                print("------")
                print("BUY")
                self.time_between_sell = self.iter
                self.buy_price = self.data.Close[self.iter]
                print(self.buy_price)
                print("------")
                self.state = self.HOLD
        if action == 2:
            if self.state is self.HOLD:
                print("------")
                print("SELL")
                self.state = self.NO_HOLD
                if math.isclose(self.buy_price, 0):
                    print("Fuck price")
                    print(self.iter)
                    reward = 0
                else:
                    reward = (self.data.Open[self.iter] - self.buy_price) / self.buy_price
                print("sell price . " + str(self.data.Open[self.iter]))
                self.sellnum = self.sellnum + 1
                print("reward: " + str(reward))
                print("sells: " + str(self.sellnum))
                print("------")
                self.cum_rewarsd = self.cum_rewarsd + reward


        obsrew = self.calc_reward()
        obs = np.array([self.data.c_o[self.iter], self.data.c_l_h[self.iter], self.data.o_c_1[self.iter],
                        self.data.aggr_mean[self.iter], self.data.aggr_std[self.iter], obsrew], dtype=float)
        norm_obs = np.linalg.norm(obs)
        if norm_obs is 0:
            norm_obs = 1
            print(obs)
            print(self.iter)

        obs = obs / norm_obs

        if math.isnan(obs[0]):
            norm_obs = 1
            print(obs)
            print(self.iter)
            obs = np.array([self.data.c_o[self.iter], self.data.c_l_h[self.iter], self.data.o_c_1[self.iter],
                            self.data.aggr_mean[self.iter], self.data.aggr_std[self.iter], obsrew], dtype=float)
        if math.isnan(reward):
            print("Reward is a fucking Nan you scam bitch fuck")
        return obs, reward, done, {}

    def reset(self):
        print("Reste")
        self.iter = 0
        self.cum_rewarsd = 0
        self.state = self.NO_HOLD
        reward = 0
        obs = np.array([self.data.c_o[self.iter], self.data.c_l_h[self.iter], self.data.o_c_1[self.iter],
                        self.data.aggr_mean[self.iter], self.data.aggr_std[self.iter], reward], dtype=float)
        return obs

    def close(self):
        return

debug = StockMarketEnv()
asd = make_vec_env(StockMarketEnv, n_envs=8)
# 512
#model = PPO2(MlpLstmPolicy, asd, n_steps=1024, gamma=0.999
#             , verbose=2, tensorboard_log="c:\\tmp\\roxenv_sh\\traning_data\\",
#             policy_kwargs={'n_lstm': 64})
#del model

model = PPO2.load("c:\\tmp\\roxenv_sh\\traning_data\\kockblocasdk_9")


obs = asd.reset()
_state = None
for i in range(10000):
    action, _state = model.predict(obs, state=_state, deterministic=True)
    prob = model.action_probability(obs)
    if action[0] == 2:
        print("Seelllll")
        print(prob[0])
    obs, reward, dones, info = asd.step(action)
    if action[0] == 2:
        print("reward: " + str(reward[0]))

