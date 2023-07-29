# 학습하는 개체, Investor

import numpy as np
import utils

class agent:
    charge = 0.00015 # 수수료
    tax = 0.0025 # 세금

    BUY=0
    SELL=1
    HOLD=2
    action = [BUY, SELL, HOLD]

    def __init__(self, environment, min_rate=0.05, max_rate=0.2, positive_threshold=0.03, negative_threshold=0.03, init_cash=0):
        self.environment = environment
        
        self.min_rate = min_rate # 자본 대비 단일 매매 최소비율
        self.max_rate = max_rate # 자본 대비 단일 매매 최대비율
        
        self.positive_threshold = threshold
        self.negative_threshold = threshold

        self.balance = dict()
        self.balance['cash'] = init_cash
        self.balance['n_stocks'] = 0
        
        self.init_pv = init_cash
        self.base_pv = init_pv
        self.pv = self.init_pv

        self.action_count=[0,0,0]
        self.stock_rate = 0
    
    def set_balance(self, balance=None, cash=None):
        if balance is not None:
            self.balance = balance
        elif cash is not None:
            self.balance['cash'] = cash
            self.balance['n_stocks'] = 0
        else:
            self.balance['cash'] = 0
            self.balance['n_stocks'] = 0

    def reset(self):
        set_balance()
        
        self.action_count = [0,0,0]
        self.base_pv = self.init_pv
        self.pv = self.init_pv
        self.stock_rate = 0

    def calculate_pv(self): # portfolio value
        self.pv = self.balance['cash'] + self.balance['n_stocks'] * self.environment.price()
        return self.pv

    def state(self):
        self.stock_rate = (self.pv - self.balance['cash']) / self.pv
        pv_rate = self.pv / self.init_pv
        return (self.stock_rate, pv_rate) # (자산 중 주식이 차지하는 비율, 전체 손익률)
    
    def action(self, epsilon, predict=None):
        action = -1
        confidence = 0.33 # default confidence

        if np.random.rand() < epsilon:
            action = self.action[np.random.randint(0, high=3)]
        else:
            action = np.argmax(predict)
            confidence = max(predict)
        
        rate = self.min_rate + (self.max_rate - self.min_rate) * confidence
        n_stocks = self.pv * rate // self.environment.price()
        return action, n_stocks
    
    def validate_action(self, action):
        if action == self.BUY:
            return self.balance['cash'] >= self.environment.price()
        elif action == self.SELL:
            return self.balance['n_stocks'] >= 0
        else:
            return True
    
    def step(self, action, n_stocks):
        if validate_action(action):
            pr_by_one = self.environment.price()
            if action == self.BUY:
                pr_by_one *= (1 + charge)
                n_stocks = min(self.balance['cash'] // pr_by_one, n_stocks)
                self.balance['cash'] -= pr_by_one * n_stocks
                self.balance['n_stocks'] += n_stocks
            elif action == self.SELL:
                pr_by_one *= (1 - (charge + tax))
                n_stocks = min(self.balance['n_stocks'], n_stocks)
                self.balance['cash'] += pr_by_one * n_stocks
                self.balance['n_stocks'] -= n_stocks
            self.action_count[action] += 1
        
        yday_pv = self.pv # 어제 자산
        pv = self.calculate_pv() # 현재 자산
        one_day_rate = (pv - yday_pv) / yday_pv # 하루 손익률
        baseline_rate = (pv - self.base_pv) / self.base_pv # threshold 넘은 이후 손익률

        if baseline_rate > self.positive_threshold or baseline_rate < -self.negative_threshold:
            self.base_pv = pv
        else:
            baseline_rate = 0
        
        return one_day_rate, baseline_rate # baseline_rate이 0이 아니라면 임계치 초과 발생하였으므로 학습 수행, one_day_rate은 학습 데이터로 쌓기?
