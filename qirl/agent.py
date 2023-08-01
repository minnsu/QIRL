import numpy as np
import tensorflow as tf
import keras

from utils import softmax

class Agent:

    def __init__(self, environment, model, optimizer, loss_fn, discount_factor=0.95, init_epsilon=0.3):
        self.environment = environment
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.discount_factor = discount_factor
        self.epsilon = init_epsilon

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

        # epsilon = self.epsilon
        n_steps = len(self.environment.chart_data)
        # epsilon_delta = self.epsilon / n_steps
        done = False
        next_state = self.environment.present()
        while done is False:
            target = np.array([0, 0, 0], dtype=np.float32)
            target_idx = [0, 1, 2]
            
            with tf.GradientTape() as tape:
                # if np.random.rand() < epsilon:
                #     prediction = softmax(np.random.rand(3))
                # else:
                #     prediction = self.model(next_state)
                prediction = self.model(next_state)
                action = (np.argmax(prediction), np.max(prediction))
                next_state, reward, done, info = self.environment.step(action)
                if reward > 0:
                    target[action[0]] = 1
                else: # reward가 음수면 다른 2가지 action에 0.5씩 제공
                    del target_idx[action[0]]
                    target[target_idx] = 0.5
                loss = self.loss_fn(target.reshape(-1, 3), prediction)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # epsilon -= epsilon_delta
        print(info)

    def train(self, n_episodes=1):
        for i in range(n_episodes):
            print('Episode {} ...'.format(i+1), end='', flush=True)
            self.run_one_episode()
            print('Done.')

    
    def predict(self, chart):
        chart = np.array(chart)
        prediction = self.model(chart.reshape(-1, len(chart)))
        return (np.argmax(prediction), np.max(prediction))