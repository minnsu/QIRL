import numpy as np

class Environment:
    tax = 0.002
    charge = 0.00015

    def __init__(self, chart_data, price_idx, init_cash, actions=[0, 0, 0], min_rate=0.05, max_rate=0.2):
        self.chart_data = chart_data
        self.price_idx = price_idx
        
        self.init_cash = init_cash
        
        self.cash = init_cash
        self.n_stocks = 0
        self.pv = self.init_cash # step마다 present balance 갱신
        
        self.actions = actions # 순서대로 buy, sell, hold의 확률
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.idx = 0
        self.state = self.chart_data[self.idx]
    
    def reset(self):
        self.cash = self.init_cash
        self.n_stocks = 0
        self.pv = self.init_cash

        self.idx = 0
        self.state = self.chart_data[self.idx]

    def present(self):
        return self.state.reshape(-1, len(self.state))

    def step(self, action): # action = (action, confidence)
        info = dict()
        done = False
        reward = 0

        price = self.state[self.price_idx]
        action, confidence = action

        n_stocks = self.min_rate + (self.max_rate - self.min_rate) * confidence
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
            info['result'] = self.pv / self.init_cash # 최종 손익률
            self.reset()
        else:
            self.state = self.chart_data[self.idx]
        
        price = self.state[self.price_idx]
        new_pv = self.cash + price * self.n_stocks
        reward = new_pv / self.pv - 1
        self.pv = new_pv
        
        return self.state.reshape(-1, len(self.state)), reward, done, info