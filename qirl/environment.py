import numpy as np
import torch
import utils

class DeepQNetworkEnvironment:
    tax = 0.002
    charge = 0.00015

    def __init__(self, chart_data, price_idx, reward_scaler=100):
        self.chart_data = chart_data
        self.price_idx = price_idx
        self.reward_scaler = reward_scaler

        self.records = [[0, 0., 0.], [0, 0., 0.]]
        
        self.idx = 0
        self.state = self.chart_data[self.idx]
    
    def reset(self):
        self.records = [[0, 0., 0.], [0, 0., 0.]]
        self.idx = 0
        self.state = self.chart_data[self.idx]
    
    def present(self):
        return self.state[np.newaxis]

    def step(self, action):
        info = dict()
        done = False

        present_price = self.state[self.price_idx]
        
        self.idx += 1
        if self.idx >= len(self.chart_data):
            done = True
            info['n_buys'] = self.records[0][0]
            info['buy_profit'] = self.records[0][1]
            info['buy_loss'] = self.records[0][2]
            info['n_sells'] = self.records[1][0]
            info['sell_profit'] = self.records[1][1]
            info['sell_loss'] = self.records[1][2]
            self.reset()
        else:
            self.state = self.chart_data[self.idx]
        next_price = self.state[self.price_idx]

        reward = 0
        if action == 0:
            reward = next_price - present_price * (1 + self.charge)
        elif action == 1:
            reward = (present_price * (1 - self.tax - self.charge)) - next_price

        self.records[action][0] += 1
        if reward > 0:
            self.records[action][1] += reward
        else:
            self.records[action][2] += reward

        return self.state, reward * self.reward_scaler, done, info


class PolicyGradientEnvironment:
    tax = 0.002
    charge = 0.00015

    def __init__(self, chart_data, price_idx, init_cash=10000000, min_rate=0.05, max_rate=0.2, hold_interval=0.1, reward_scaler=100):
        self.chart_data = utils.preprocess(chart_data, False)
        self.train_data = utils.preprocess(chart_data)
        self.price_idx = price_idx
        
        self.init_cash = init_cash
        self.cash = self.init_cash
        self.n_stocks = 0
        self.pv = self.init_cash
        self.records = [[0, 0., 0.], [0, 0., 0.], [0, 0., 0.]] # [n_buys, buy_profit, buy_loss], [n_sells, sell_profit, sell_loss], [n_holds, hold_profit, hold_loss]

        self.min_rate = min_rate
        self.max_rate = max_rate
        
        self.hold_interval = hold_interval
        self.reward_scaler = reward_scaler

        self.idx = 0
        self.train_state = self.train_data[self.idx]
        self.state = self.chart_data[self.idx]
    
    def reset(self):
        self.cash = self.init_cash
        self.n_stocks = 0
        self.pv = self.init_cash
        self.records = [[0, 0., 0.], [0, 0., 0.], [0, 0., 0.]]

        self.idx = 0
        self.train_state = self.train_data[self.idx]
        self.state = self.chart_data[self.idx]
    
    def present(self, training=True):
        return self.train_state if training else self.state
    
    def step(self, action): # (action, confidence)
        info = dict()
        done = False
        reward = 0

        price = self.state[self.price_idx]
        new_pv = self.cash + price * self.n_stocks
        reward = (new_pv / self.pv) - 1
        self.pv = new_pv
        
        action, confidence = action
        if confidence < 0.5 + self.hold_interval / 2 and confidence > 0.5 - self.hold_interval:
            action = 2
        
        self.records[action][0] += 1
        if reward > 0:
            self.records[action][1] += float(reward)
        else:
            self.records[action][2] += float(reward)
        
        n_stocks = (self.min_rate + (self.min_rate + self.max_rate) * confidence) * self.pv // price
        if action == 0: # buy
            n_stocks = min(n_stocks, self.cash // price)
            self.n_stocks += n_stocks
            self.cash -= n_stocks * (price * (1 + self.charge))
        elif action == 1: # sell
            n_stocks = min(n_stocks, self.n_stocks)
            self.n_stocks -= n_stocks
            self.cash += n_stocks * (price * (1 - self.tax - self.charge))
        else: # hold
            pass
        
        self.idx += 1
        if self.idx >= len(self.chart_data):
            done = True
            info['final_profit'] = float(self.pv) / self.init_cash # 최종 손익률
            info['[buy, profit, loss]'] = [self.records[0][0], self.records[0][1], self.records[0][2]]
            info['[sell, profit, loss]'] = [self.records[1][0], self.records[1][1], self.records[1][2]]
            info['[hold, profit, loss]'] = [self.records[2][0], self.records[2][1], self.records[2][2]]
            self.reset()
        else:
            self.state = self.chart_data[self.idx]
            self.train_state = self.train_data[self.idx]

        return self.train_state, reward * self.reward_scaler, done, info # reward scale 키우기 위해 *100

class ActorCriticEnvironment:
    tax = 0.002
    charge = 0.00015
    
    def __init__(self, chart_data, price_idx, reward_scaler=100):
        self.chart_data = chart_data
        self.price_idx = price_idx
        self.length = len(self.chart_data)

        self.records = [[0, 0., 0.], [0, 0., 0.]] # [Buys, profit, loss], [Sells, profit, loss]

        self.idx = 0
        self.state = self.chart_data[self.idx]
        
        self.reward_scaler = reward_scaler
        pass

    def reset(self):
        self.records = [[0, 0., 0.], [0, 0., 0.]] # [Buys, profit, loss], [Sells, profit, loss]
        self.idx = 0
        self.state = self.chart_data[self.idx]

    def present(self):
        return self.state

    def step(self, action):
        info = dict()
        reward = 0
        done = False

        price = self.state[self.price_idx]

        self.idx += 1
        self.state = self.chart_data[self.idx]
        next_price = self.state[self.price_idx]
        
        reward = 0
        if action == 0:
            reward = next_price - price * (1 + self.charge)
        elif action == 1:
            reward = (price * (1 - self.tax - self.charge)) - next_price

        self.records[action][0] += 1
        if reward > 0:
            self.records[action][1] += reward
        else:
            self.records[action][2] += reward
        
        if self.idx >= len(self.chart_data) - 1:
            done = True
            info['Buys'] = [self.records[0][0], self.records[0][1], self.records[0][2]]
            info['Sells'] = [self.records[1][0], self.records[1][1], self.records[1][2]]
            self.reset()
        
        return self.state, reward * self.reward_scaler, done, info
        