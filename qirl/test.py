import utils
import numpy as np
import torch

from agent import AdvancedAgent
from environment import AdvancedEnvironment

# Preprocess
# utils.preprocess('./data/prep1_1.pkl', './data/rl_train1.pkl')

# Load data
chart_data = utils.load_data('./data/prep1_1.pkl')

# Model design
from test_model import TestModel

n_actions = 2
env = AdvancedEnvironment(chart_data[0], 4, hold_interval=0.06)

model = TestModel(9, 2)

optimizer = torch.optim.Adam(model.model.parameters())
loss_fn = torch.nn.functional.mse_loss
agent = AdvancedAgent(env, model, optimizer, loss_fn)

model = torch.load('./model/test_advanced.keras')
agent.set_model(model)
# agent.train(n_episodes=30)
torch.save(model, './model/test_advanced.keras')

env.reset()
for idx, state in enumerate(chart_data[0].swapaxes(0, 1)[:, :, 3]):
    action = agent.predict(state)
    next_state, rewards, done, info = env.step(action)
    if done:
        print(info)
        break