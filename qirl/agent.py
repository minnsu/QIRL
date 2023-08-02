import numpy as np
import tensorflow as tf
import keras

from utils import softmax

class PolicyGradientAgent:

    def __init__(self, environment, model, optimizer, loss_fn, discount_factor=0.95):
        self.environment = environment
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.discount_factor = discount_factor

    def set_model(self, model):
        self.model = model

    def discount_rewards(self, rewards):
        discounted = np.array(rewards)
        for step in range(len(rewards) -2, -1, -1):
            discounted[step] += discounted[step+1] * self.discount_factor
        return discounted

    def discount_normalize_rewards(self, all_rewards):
        all_discounted = [self.discount_rewards(rewards) for rewards in all_rewards]

        flattened = np.concatenate(all_discounted)
        reward_mean = flattened.mean()
        reward_std = flattened.std()

        return [(discounted - reward_mean) / reward_std for discounted in all_discounted]

    def run_one_episode(self):
        done = False
        next_state = self.environment.present()
        while done is False:
            target = np.array([0, 0, 0], dtype=np.float32)
            target_idx = [0, 1, 2]
            
            with tf.GradientTape() as tape:
                prediction = self.model(next_state)
                action = (np.argmax(prediction), np.max(prediction))
                next_state, reward, done, info = self.environment.step(action)
                next_state = next_state[np.newaxis]
                if reward > 0: # reward 크기에 따른 target 조정?
                    target[action[0]] = 1
                else: # reward가 음수면 다른 2가지 action에 0.5씩 제공
                    del target_idx[action[0]]
                    target[target_idx] = 0.5
                loss = self.loss_fn(target.reshape(-1, 3), prediction)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        print(info)

    def train(self, n_episodes=1):
        for i in range(n_episodes):
            print('Episode {} ...'.format(i+1), end='', flush=True)
            self.run_one_episode()
            print('Done.')

    def predict(self, state):
        prediction = self.model(state[np.newaxis])
        return (np.argmax(prediction), np.max(prediction))

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
    
    def set_model(self, model):
        self.model = model
    
    def get_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions) # action 종류
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose=0)
            return np.argmax(Q_values[0])

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_idx] for experience in batch])
            for field_idx in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, state, epsilon):
        action = self.get_action(state, epsilon)
        next_state, reward, done, info = self.environment.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        # 최대 Q 가치 찾기
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1-dones) * self.discount_factor * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.n_actions)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

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
                next_state, reward, done, info = self.play_one_step(next_state, epsilon)
                if done:
                    print(info)
                    break
                epsilon -= epsilon_delta
            self.training_step(batch_size=n_steps)
            print('Done.')
    
    def predict(self, state):
        Q_values = self.model(state[np.newaxis])
        return np.argmax(Q_values)