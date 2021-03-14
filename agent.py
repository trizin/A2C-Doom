# %% imports

import numpy as np
import tensorflow as tf
from collections import deque
from skimage import transform
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
# %% Actor model
eps = np.finfo(np.float32).eps.item()
tf.random.set_seed(84)
np.random.seed(84)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
state_size = [84, 84, 4]
stack_size = 4
shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]


def preprocess_frame(frame):
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


def preprocess_frames(frames):
    preprocessed = [preprocess_frame(i) for i in frames]
    return np.stack(preprocessed, axis=2).reshape(1, 84, 84, stack_size)


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.conv1 = layers.Conv2D(32, 8, 4, padding="same", input_shape=((84, 84, stack_size)), kernel_initializer=tf.keras.initializers.glorot_normal())
        self.activation = layers.ELU()
        self.conv2 = layers.Conv2D(64, 4, 2, padding="same", kernel_initializer=tf.keras.initializers.glorot_normal())
        self.conv3 = layers.Conv2D(128, 2, 2, padding="valid", kernel_initializer=tf.keras.initializers.glorot_normal())
        self.normalization = layers.BatchNormalization(epsilon=1e-5, name="batch_norm")
        self.normalization1 = layers.BatchNormalization(epsilon=1e-5, name="batch_norm")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.conv1(inputs)
        x = self.normalization(x)
        X = self.activation(x)

        x = self.conv2(x)
        x = self.normalization1(x)
        X = self.activation(x)

        x = self.conv3(x)
        X = self.activation(x)

        x = self.flatten(x)
        x = self.dense1(x)

        return self.actor(x), self.critic(x)

# %% Agent class


class Agent:
    def __init__(self, num_actions, optimizer, env, gamma):
        self.model = ActorCritic(num_actions)
        self.optimizer = optimizer
        self.env = env
        self.gamma = gamma

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print("Model loaded "+path)

    def env_step(self, action):
        reward = self.env.make_action(actions[action])
        done = self.env.is_episode_finished()
        if done:
            return None, reward, done
        state = self.getinitialState()
        return state, reward, done

    def getinitialState(self):
        return tf.constant(self.env.get_state().screen_buffer, dtype=tf.float32)

    def run_episode(self, max_steps=50000):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.env.new_episode()
        state = self.getinitialState()
        next_state = state
        stacked_frames = deque([state for i in range(stack_size)], maxlen=stack_size)
        for t in tf.range(max_steps):
            if self.env.is_episode_finished():
                break
            action_logits_t, value = self.model(preprocess_frames(stacked_frames))
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            values = values.write(t, tf.squeeze(value))

            action_probs = action_probs.write(t, action_probs_t[0, action])

            next_state, reward, done = self.env_step(action.numpy())
            stacked_frames.append(next_state)

            rewards = rewards.write(t, reward)

            if done == True:
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards, standardize=True):
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + np.finfo(np.float32).eps.item()))

        return returns

    def compute_loss(self, action_probs, values, returns):
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss

    def train_step(self, max_steps_per_episode):

        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.run_episode(max_steps_per_episode)
            returns = self.get_expected_return(rewards)
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
            loss = self.compute_loss(action_probs, values, returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward
