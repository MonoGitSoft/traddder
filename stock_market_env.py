import gym
import numpy as np
import pandas as pd
import math
from gym import spaces
from stable_baselines.common.policies import  MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

class StockMarketEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    # (sort, long, idle) or (sort, long)

    num_of_action = 3
    date = 'Date'
    open = 'Open'
    close = 'Close'
    high = 'High'
    low = 'Low'
    new_stock = 'new_stock'

    #FEATURES
    c_o = 'c_o'
    c_l_h = 'c_l_h'
    o_c_1 = 'o_c_1'
    aggr_mean = 'aggr_mean'
    aggr_std = 'aggr_std'

    #state
    HOLD = 'Hold'
    NO_HOLD = 'Not hold'
    def __init__(self):
        super(StockMarketEnv, self).__init__()
        self.action_space = spaces.Discrete(self.num_of_action)
        self.observation_space = spaces.Box(low = -1, high=1,shape=(1,6), dtype=float)
        self.data_frame = pd.read_csv("traning_data\\traning_features.csv")
        self.balance = 10000
        self.hold = False
        self.buy_price = 0
        self.iter = 0
        self.data = pd.read_csv("c:\\tmp\\roxenv_sh\\traning_data\\traning_features.csv")
        self.state = self.NO_HOLD
        self.cum_rewarsd = 0

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
        self.iter = self.iter + 1
        if self.iter == 10000: #2340390
            print("Reward:" + str(self.cum_rewarsd))
            done = True
            self.iter = 0
        if self.data.new_stock[self.iter] == 0:
            print("New stock")
            action = 2

        #if action == 0:

        if action == 1:
            if self.state is self.NO_HOLD:
                #print("BUY")
                self.buy_price = self.data.Close[self.iter]
                self.state = self.HOLD
        if action == 2:
            if self.state is self.HOLD:
                #print("SELL")
                self.state = self.NO_HOLD
                if math.isclose(self.buy_price, 0):
                    print("Fuck price")
                    print(self.iter)
                    reward = 0
                else:
                    reward = (self.data.Open[self.iter] - self.buy_price) / self.buy_price
                self.cum_rewarsd = self.cum_rewarsd + reward

        done = False

        reward = self.calc_reward()

        obs = np.array([self.data.c_o[self.iter], self.data.c_l_h[self.iter], self.data.o_c_1[self.iter],
                        self.data.aggr_mean[self.iter], self.data.aggr_std[self.iter], reward], dtype=float)
        norm_obs = np.linalg.norm(obs)
        if  norm_obs is 0:
            norm_obs = 1
            print(obs)
            print(self.iter)

        obs = obs/norm_obs

        if math.isnan(obs[0]):
            norm_obs = 1
            print(obs)
            print(self.iter)
            obs = np.array([self.data.c_o[self.iter], self.data.c_l_h[self.iter], self.data.o_c_1[self.iter],
                        self.data.aggr_mean[self.iter], self.data.aggr_std[self.iter], reward], dtype=float)
        if math.isnan(reward):
            print("Reward is a fucking Nan you scam bitch fuck")

        self.iter = self.iter + 1
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


asd = make_vec_env(StockMarketEnv, n_envs=4)
print(asd)

model = PPO2(MlpLstmPolicy, asd, n_steps=2048,verbose=2, tensorboard_log="c:\\tmp\\roxenv_sh\\traning_data\\",policy_kwargs={'n_lstm': 32})

model.learn(total_timesteps=(2048 * 4 * 10000)
            )

obs = asd.reset()

for i in range(10000):
    action, _state = model.predict(obs)
    obs, reward, dones, info = asd.step(action)
    print(reward)
