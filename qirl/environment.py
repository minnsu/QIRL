import numpy as np

class Environment:
    tax = 0.002
    charge = 0.00015

    def __init__(self, chart_data, price_idx, init_cash, min_rate=0.05, max_rate=0.2):
        self.chart_data = chart_data
        self.price_idx = price_idx
        
        self.init_cash = init_cash
        
        self.cash = init_cash
        self.n_stocks = 0
        self.pv = self.init_cash # step마다 present balance 갱신
        
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.act_profit = [[0, 0.], [0, 0.], [0, 0.]] # [n_buys, buy_profit], [n_sells, sell_profit], [n_holds, hold_profit]

        self.idx = 0
        self.state = self.chart_data[self.idx]
    
    def reset(self):
        self.cash = self.init_cash
        self.n_stocks = 0
        self.pv = self.init_cash

        self.act_profit = [[0, 0.], [0, 0.], [0, 0.]]

        self.idx = 0
        self.state = self.chart_data[self.idx]

    def present(self):
        return self.state[np.newaxis]

    def step(self, action): # action = (action, confidence)
        info = dict()
        done = False
        reward = 0

        price = self.state[self.price_idx]
        new_pv = self.cash + price * self.n_stocks
        reward = new_pv / self.pv - 1
        self.pv = new_pv
        self.act_profit[action][0] += 1
        self.act_profit[action][1] += reward
        
        # Policy Gradient
        # action, confidence = action
        # n_stocks = self.min_rate + (self.max_rate - self.min_rate) * confidence

        # Deep Q Network
        n_stocks = (self.min_rate + self.max_rate) / 2
        if action == 0: # buy
            n_stocks = min(self.pv * n_stocks // price, self.cash // price)
            self.n_stocks += n_stocks
            self.cash -= n_stocks * (price * (1 + self.charge))
        elif action == 1: # sell
            self.cash += self.n_stocks * (price * (1 - self.tax - self.charge))
            self.n_stocks = 0
        else: # hold
            pass
        
        
        price = self.state[self.price_idx]
        
        self.idx += 1
        if self.idx >= len(self.chart_data):
            done = True
            info['final_profit'] = self.pv / self.init_cash # 최종 손익률
            info['n_buys'] = self.act_profit[0][0]
            info['buy_profit'] = self.act_profit[0][1]
            info['n_sells'] = self.act_profit[1][0]
            info['sell_profit'] = self.act_profit[1][1]
            info['n_holds'] = self.act_profit[2][0]
            info['hold_profit'] = self.act_profit[2][1]
            self.reset()
        else:
            self.state = self.chart_data[self.idx]

        return self.state, reward * 100, done, info # reward scale 키우기 위해 *100

class StockEnvironment:
    tax = 0.002
    charge = 0.00015

    def __init__(self, chart_data, price_idx, reward_scaler=100):
        self.chart_data = chart_data
        self.price_idx = price_idx
        self.reward_scaler = reward_scaler

        self.act_profit = [[0, 0., 0.], [0, 0., 0.]]
        
        self.idx = 0
        self.state = self.chart_data[self.idx]
    
    def reset(self):
        self.act_profit = [[0, 0., 0.], [0, 0., 0.]]
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
            info['n_buys'] = self.act_profit[0][0]
            info['buy_profit'] = self.act_profit[0][1]
            info['buy_loss'] = self.act_profit[0][2]
            info['n_sells'] = self.act_profit[1][0]
            info['sell_profit'] = self.act_profit[1][1]
            info['sell_loss'] = self.act_profit[1][2]
            self.reset()
        else:
            self.state = self.chart_data[self.idx]
        next_price = self.state[self.price_idx]

        reward = 0
        if action == 0:
            reward = next_price - present_price
        elif action == 1:
            reward = present_price - next_price

        self.act_profit[action][0] += 1
        if reward > 0:
            self.act_profit[action][1] += reward
        else:
            self.act_profit[action][2] += reward

        return self.state, reward * self.reward_scaler, done, info
