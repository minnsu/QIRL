import utils
import numpy as np
import torch

from agent import ActorCriticAgent
from environment import ActorCriticEnvironment

# Preprocess
# prep1 = utils.load_data('./data/prep1_1.pkl')
# training = []
# for stock in prep1:
#     tmp = utils.preprocess(stock)
#     training.append(tmp)
# utils.store_data('./data/train.pkl', training)

# Load data
chart_data = utils.load_data('./data/train.pkl')

# Model design
from test_model import ActorCriticModel

n_actions = 2
env = ActorCriticEnvironment(chart_data[0], 4)

model = ActorCriticModel(9, 2)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.functional.mse_loss
agent = ActorCriticAgent(env, model, optimizer, loss_fn)

model = torch.load('./model/test_act_cri.keras')
agent.set_model(model)
agent.run(n_episodes=10, print_each_episode=True)
torch.save(model, './model/test_act_cri.keras')

print(model(chart_data[0][0]))
# env.reset()
# for idx, state in enumerate(chart_data[0].swapaxes(0, 1)[:, :, 3]):
#     action = agent.predict(state)
#     next_state, rewards, done, info = env.step(action)
#     if done:
#         print(info)
#         break