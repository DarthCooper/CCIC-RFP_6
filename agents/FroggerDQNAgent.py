import numpy as np
import pickle
import os
from typing import Tuple

from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, Flatten
from keras.api.optimizers import Adam
from keras.api.models import load_model
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import cv2

from envs.FroggerEnv import CustomVecEnv
from buffers.replayBuffer import ReplayBuffer

from configs.FroggerConfig import BuildFroggerConfig as BFConfig
from configs.FroggerConfig import MemoryFroggerConfig as MFConfig
from configs.FroggerConfig import PreprocessingFroggerConfig as PFConfig
from configs.FroggerConfig import ModelConfig as MConfig

class DQNAgent:
    def __init__(self, resume:bool = False, env=CustomVecEnv):
        self.resume = resume

        self.SAVE_DIR = MFConfig.save_dir
        self.MODEL_PATH = os.path.join(MFConfig.save_dir, MFConfig.model_path)
        self.MEMORY_PATH = os.path.join(MFConfig.save_dir, MFConfig.memory_path)
        self.save_freq = MFConfig.save_freq

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        self.env = env

        self.stack_size = 4
        self.frame_stack = [deque(maxlen=self.stack_size) for _ in range(self.env.get_num_envs())]

        self.frame_size = PFConfig.screen_size
        self.grayscale = PFConfig.grayscale_obs

        self.buffer = ReplayBuffer()
        self.episodes = MConfig.episodes
        self.epsilon = MConfig.epsilon
        self.epsilon_min = MConfig.epsilon_min
        self.epsilon_decay = MConfig.epsilon_decay
        self.gamma = MConfig.gamma
        self.batch_size = MConfig.batch_size
        self.update_target_freq = MConfig.update_target_freq
        self.max_steps = BFConfig.max_episode_steps

        self.obs_shape = (*self.frame_size, self.stack_size)
        self.n_actions = env.action_space.n

        self.model = self._load_or_create_model()
        self.target_model = self.create_dqn_model(self.obs_shape, self.n_actions)
        self.target_model.set_weights(self.model.get_weights())

        self.step_count = 0
        self.episode = 1
        self.all_rewards = []


        if self.resume:
            self._load_memory()

    def preprocess(self, obs : np.ndarray):
        if len(obs.shape) == 3 and obs.shape[-1] == 3 and self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs - cv2.resize(obs, self.frame_size, interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0

        if self.grayscale and len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=-1)
        return obs
    
    def stack_frames(self, frame_list : np.ndarray, is_new_episode :bool):
        stacked = []
        for i, frame in enumerate(frame_list):
            if is_new_episode:
                self.frame_stack[i].clear()
                for _ in range(self.stack_size):
                    self.frame_stack[i].append(frame)
            else:
                self.frame_stack[i].append(frame)

            stacked_frame = np.concatenate(list(self.frame_stack[i]), axis=-1)
            stacked.append(stacked_frame)

        return stacked

    def train(self):
        for episode in range(1, self.episodes + 1):
            obs_batch : np.ndarray = self.env.reset()

            for i in range(0, 100):
                next_obs_batch, rewards, dones_batch, truncateds_batch, _ = self.env.step([1] * self.env.get_num_envs())
                if (all(dones_batch) or all(truncateds_batch)):
                    break

            obs_batch = [self.preprocess(obs) for obs in obs_batch]
            obs_batch = self.stack_frames(obs_batch, is_new_episode=True)

            dones = [False] * self.env.get_num_envs()
            dead = [False] * self.env.get_num_envs()
            truncateds = [False] * self.env.get_num_envs()
            total_rewards = [0] * self.env.get_num_envs()

            for step in range(self.max_steps):
                actions = [self.epsilon_greedy_action(obs, self.model, self.n_actions, self.epsilon) for obs in obs_batch]
                next_obs_batch, rewards, dones_batch, truncateds_batch, info_batch = self.env.step(actions)
                next_obs_batch = [self.preprocess(obs) for obs in next_obs_batch]
                next_obs_batch = self.stack_frames(next_obs_batch, is_new_episode=False)

                for i in range(self.env.get_num_envs()):
                    self.buffer.add((obs_batch[i], actions[i], rewards[i], next_obs_batch[i], dones_batch[i]))
                    total_rewards[i] += rewards[i]
                    dead[i] = info_batch[i]['lives'] == 0

                if(all(dead)):
                    print("All died...loading next episode")
                    dones_batch = [True] * self.env.get_num_envs()

                obs_batch = next_obs_batch
                dones = dones_batch
                truncateds = truncateds_batch

                if all(dones) or all(truncateds):
                    break

                if len(self.buffer) >= self.batch_size:
                    states_s, actions_s, rewards_s, next_states_s, dones_s = self.buffer.sample(self.batch_size)

                    q_targets = self.model.predict(states_s, verbose=0)
                    q_next = self.target_model.predict(next_states_s, verbose=0)
                    max_q_next = np.max(q_next, axis=1)

                    for i in range(self.batch_size):
                        if(dones_s[i]):
                            q_targets[i][actions_s[i]] = rewards_s[i]
                        else:
                            q_targets[i][actions_s[i]] = rewards_s[i] + (1 - dones_s[i]) * self.gamma * max_q_next[i]

                    self.model.fit(states_s, q_targets, epochs=1, verbose=0)

                    self.step_count += 1
                    if self.step_count % self.update_target_freq == 0:
                        self.target_model.set_weights(self.model.get_weights())

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                avg_reward = np.mean(total_rewards)
                self.all_rewards.append(avg_reward)

                print(f"Episode {self.episode} - Avg Reward: {avg_reward:.2f} - Epsilon: {self.epsilon:.4f} - Steps {self.step_count} - infos : {info_batch}")

            self.episode += 1

            if episode % self.save_freq == 0:
                self._save_progress()

        self.env.close()

    def test(self, episodes=5, render=False, record=False):
        test_env : gym.Env = gym.make("ALE/Frogger-v5", render_mode="human")
        test_env = AtariPreprocessing(test_env, frame_skip=1, noop_max=0)
        
        print(f"[TEST] Running {episodes} test episodes...")

        for ep in range(episodes):
            obs, info = test_env.reset()
            obs = self.preprocess(obs)
            self.frame_stack[0].clear()
            for _ in range(self.stack_size):
                self.frame_stack[0].append(obs)
            state = np.concatenate(list(self.frame_stack[0]), axis=-1)

            done = False
            total_reward = 0
            frames = []

            while not done:
                action = np.argmax(self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
                next_obs, reward, done, _, _ = test_env.step(action)
                total_reward += reward

                if render:
                    frame = test_env.render()
                    frames.append(frame)

                next_obs = self.preprocess(next_obs)
                self.frame_stack[0].append(next_obs)
                state = np.concatenate(list(self.frame_stack[0]), axis=-1)

            print(f"[TEST] Episode {ep + 1} reward: {total_reward}")

            if record:
                os.makedirs("test_videos", exist_ok=True)
                out = cv2.VideoWriter(f'test_videos/episode_{ep+1}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()

        test_env.close()
            

    def create_dqn_model(self, input_shape : Tuple[int, int, int] = (84, 84, 1), num_actions : int = 5) -> Sequential:
        """Build a CNN model for Deep Q-Learning"""
        model = Sequential([
            Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=input_shape),
            Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(num_actions, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
        return model
    
    def epsilon_greedy_action(self, state, model : Sequential, num_actions : int, epsilon : float):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])
    
    def _load_or_create_model(self):
        model = self.create_dqn_model(self.obs_shape, self.n_actions)
        weights_path = os.path.join(self.SAVE_DIR, "dqn_weights.weights.h5")
        if self.resume and os.path.exists(weights_path):
            print("[INFO] Loading model weights...")
            model.load_weights(weights_path)
        return model
        
    def _load_memory(self):
        if os.path.exists(self.MEMORY_PATH):
            with open(self.MEMORY_PATH, "rb") as f:
                memory = pickle.load(f)
                self.episode = memory.get("episode", 0)
                self.epsilon = memory.get("epsilon", 1.0)
                self.all_rewards = memory.get("all_rewards", [])
                print(f"[INFO] Resumed from episode {self.episode}")

    def _save_progress(self):
        weights_path = os.path.join(self.SAVE_DIR, "dqn_weights.weights.h5")
        self.model.save_weights(weights_path)
        with open(self.MEMORY_PATH, "wb") as f:
            pickle.dump({
                "episode": self.episode,
                "epsilon": self.epsilon,
                "all_rewards": self.all_rewards
            }, f)
        print(f"[Checkpoint] Episode {self.episode} saved.")
