import numpy as np
import torch
import torch.nn.functional as F

from utils import softmax

class DeepQNetworkAgent:
    def __init__(self, environment, model, optimizer, loss_fn, discount_factor=0.95, epsilon=0.5, n_actions=2):
        self.environment = environment
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.n_actions = n_actions
        
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        from collections import deque
        self.replay_buffer = deque(maxlen=10000)
    
    def set_env(self, env):
        self.environment = env

    def set_model(self, model):
        self.model = model
    
    def get_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions) # action 종류
        else:
            Q_values = self.model(state)
            return torch.argmax(Q_values)

    def sample_experiences(self, batch_size):
        indices = torch.randint(len(self.replay_buffer), (batch_size, ))
        batch = [self.replay_buffer[idx] for idx in indices]

        states = torch.stack(tuple([experience[0].reshape(1, -1) for experience in batch]), dim=0)
        actions = torch.tensor([experience[1] for experience in batch])
        rewards = torch.stack(tuple([experience[2].reshape(1, -1) for experience in batch]), dim=0)
        next_states = torch.stack(tuple([experience[3].reshape(1, -1) for experience in batch]), dim=0)
        dones = torch.tensor([experience[4] for experience in batch])

        # states, actions, rewards, next_states, dones = [
        #     torch.stack(tuple([experience[field_idx].reshape(1, -1) for experience in batch]), dim=0)
        #     for field_idx in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, state, epsilon):
        action = self.get_action(state, epsilon)
        next_state, reward, done, info = self.environment.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        # 최대 Q 가치 찾기
        next_Q_values = self.model(next_states)
        max_next_Q_values = next_Q_values.max(dim=2).values
        target_Q_values = (rewards + ~dones * self.discount_factor * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = F.one_hot(actions, num_classes=self.n_actions)
        
        all_Q_values = self.model(states)
        Q_values = torch.sum(all_Q_values * mask, (1, ), keepdims=True)
        loss = torch.mean(self.loss_fn(target_Q_values, Q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, n_episodes=1, n_steps=None):
        for episode in range(n_episodes):
            print('Episode {} ...'.format(episode + 1), end='', flush=True)
            epsilon = self.epsilon
            epsilon_delta = epsilon / n_episodes

            self.environment.reset()
            next_state = self.environment.present()
            if n_steps is None:
                n_steps = len(self.environment.chart_data)
            for step in range(n_steps):
                next_state = torch.tensor(next_state, dtype=torch.float32)
                next_state, reward, done, info = self.play_one_step(next_state, epsilon)
                if done:
                    print(info)
                    break
                epsilon -= epsilon_delta
            self.training_step(batch_size=32)
            print('Done.')
    
    def predict(self, state):
        Q_values = self.model(torch.tensor(state, dtype=torch.float32))
        print(Q_values)
        return torch.argmax(Q_values)

class PolicyGradientAgent:
    def __init__(self, env, model, optimizer, loss_fn, discount_factor=0.95, epsilon=0.5, n_outputs=2):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        self.n_outputs = 2

    def set_model(self, model):
        self.model = model
    
    def set_env(self, env):
        self.env = env
    
    def get_action(self, state, epsilon=0):
        if torch.rand(1) < epsilon:
            return softmax(torch.rand(self.n_outputs)) # action 종류
        else:
            conf_pair = self.model(state)
            return conf_pair
    
    def discount_rewards(self, rewards):
        discounted = torch.tensor(rewards)
        for step in range(len(rewards) -2, -1, -1):
            discounted[step] += discounted[step+1] * self.discount_factor
        return discounted

    def discount_normalize_rewards(self, all_rewards):
        all_discounted = [self.discount_rewards(rewards) for rewards in all_rewards]

        flattened = np.concatenate(all_discounted)
        reward_mean = float(flattened.mean())
        reward_std = float(flattened.std())

        return [(discounted - reward_mean) / reward_std for discounted in all_discounted]
    
    def run_one_episode(self, n_steps):
        done = False
        next_state = self.env.present()
        
        epsilon = self.epsilon
        epsilon_delta = epsilon / n_steps
        while done is False:
            target = torch.tensor([0, 0], dtype=torch.float32)
            prediction = self.get_action(next_state, epsilon)
            confidence = max(prediction)
            action = torch.argmax(prediction)
            next_state, reward, done, info = self.env.step((action, confidence))

            if reward > 0: # reward 크기에 따른 target 조정?
                target[action] = 1
            else:
                target[1-action] = 1
            
            self.optimizer.zero_grad()
            target = torch.tensor(target, dtype=torch.float32, requires_grad=True)
            loss = self.loss_fn(prediction, target)
            loss.backward()
            self.optimizer.step()

            epsilon -= epsilon_delta
        return info

    def train(self, n_episodes=1, n_steps=None):
        if n_steps == None:
            n_steps = len(self.env.chart_data)
        for i in range(n_episodes):
            print('Episode {} ...'.format(i+1), end='', flush=True)
            info = self.run_one_episode(n_steps)
            print('Done.')
            print(info)

    def predict(self, state):
        prediction = self.model(torch.tensor(state, dtype=torch.float32))
        return (int(torch.argmax(prediction)), float(torch.max(prediction)))

class ActorCriticAgent:
    def __init__(self, env, policy_value_model, optimizer, loss_fn, discount_factor=0.99, exploration_rate=None, replay_len=None):
        self.env = env
        self.model = policy_value_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.discount_factor = discount_factor
        self.exp_rate = exploration_rate

        from collections import deque
        if replay_len is None:
            replay_len = self.env.length
        self.replay = deque(maxlen=replay_len + 1)
        self.rewards = []

    def set_model(self, model):
        self.model = model
    
    def set_env(self, env):
        self.env = env

    def get_action(self, state):
        action_probs, value = self.model(state)
        tmp = torch.distributions.Categorical(action_probs)
        action = tmp.sample()
        self.replay.append((tmp.log_prob(action), value))
        return action.item()

    def after_each_episode(self):
        p_loss = []
        v_loss = []

        rewards = torch.tensor(self.rewards)
        for step in range(len(rewards)-2, -1, -1):
            rewards[step] += rewards[step+1] * self.discount_factor
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for (log_prob, value), reward in zip(self.replay, rewards):
            advantage = reward - value.item()
            p_loss.append(-log_prob * advantage)
            v_loss.append(F.smooth_l1_loss(value, reward))
        
        self.optimizer.zero_grad()
        loss = torch.stack(p_loss).sum() + torch.stack(v_loss).sum()
        loss.mean().backward()
        self.optimizer.step()

        self.rewards.clear()
        self.replay.clear()
            
    def run(self, n_episodes=1, n_steps=None, print_each_episode=False):
        if n_steps is None:
            n_steps = self.env.length
        for episode in range(n_episodes):
            print('Episode {} playing... '.format(episode+1), end='', flush=True)
            state = self.env.present()
            for step in range(n_steps):
                action = self.get_action(state)
                state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                if done:
                    if print_each_episode:
                        print(info)
                    break
            print('done.')
            print('Episode {} training... '.format(episode+1), end='', flush=True)
            self.after_each_episode()
            print('done.')