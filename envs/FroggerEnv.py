import gymnasium as gym
import numpy as np
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from stable_baselines3.common.vec_env import VecEnv
from gymnasium.wrappers import AtariPreprocessing
from queue import Empty

import ale_py

from envs.MultiActionSpace import ActionSpaces
from configs.FroggerConfig import BuildFroggerConfig as BFconfig
from configs.FroggerConfig import PreprocessingFroggerConfig as PFconfig

def worker(i, q_in : multiprocessing.Queue, q_out : multiprocessing.Queue, env : gym.Env):
    """Worker process that creates the environment and handles requests."""
    print(f"Worker {i} process started.")
    while True:
        try:
            command, data = q_in.get(timeout=1)
            print(f"Worker {i} received command: {command}")
            if command == 'reset':
                print(f"Worker {i} resetting environment...")
                obs, info = env.reset()
                q_out.put((obs, info))
            elif command == 'step':
                print(f"Worker {i} stepping environment...")
                obs, reward, done, truncated, info = env.step(data)
                q_out.put((obs, reward, done, truncated, info))
            elif command == 'close':
                print(f"Worker {i} closing environment...")
                env.close()
                break
        except Empty:
            continue

class CustomVecEnv(VecEnv):
    def __init__(self, num_envs : int = BFconfig.num_envs, render : str = BFconfig.render_mode):
        self.num_envs = num_envs
        self.envs = [None] * self.num_envs
        self.action_spaces = [None] * self.num_envs
        self.processes = []
        self.queues = []
        self.closed = False
        

        manager = multiprocessing.Manager()
        self.envs = manager.list([None] * self.num_envs)

        for i in range(self.num_envs):
            env = gym.make(
                id = BFconfig.env_name, 
                max_episode_steps = BFconfig.max_episode_steps,
                disable_env_checker = BFconfig.disable_env_checker,
                render_mode = render,
                full_action_space = BFconfig.full_action_space
                )
            env = AtariPreprocessing(
                env = env,
                noop_max = PFconfig.noop_max,
                frame_skip = PFconfig.frame_skip,
                screen_size = PFconfig.screen_size,
                terminal_on_life_loss = PFconfig.terminal_on_life_loss,
                grayscale_obs = PFconfig.grayscale_obs,
                grayscale_newaxis = PFconfig.grayscale_newaxis,
                scale_obs = PFconfig.scale_obs
                )
            
            self.action_spaces[i] = env.action_space

            q_in = multiprocessing.Queue()
            q_out = multiprocessing.Queue()
            self.queues.append((q_in, q_out))
            p = multiprocessing.Process(target=worker, args=(i, q_in, q_out, env))
            print(f"Main process starting worker {i}")
            p.start()
            self.processes.append(p)
            self.envs[i] = env

        self.action_spaces = ActionSpaces(self.action_spaces)

    def reset(self, **kwargs):
        """Reset all environments and return observations."""
        print("Main process sending reset command...", flush=True)

        for q_in, _ in self.queues:
            q_in.put(('reset', None))

        observations = []
        for _, q_out in self.queues:
            try:
                print("Main process waiting for response...", flush=True)
                obs, info = q_out.get(timeout=5)
                print(f"Main process received observation with shape: {obs.shape}", flush=True)
                observations.append(obs)
            except Empty:
                print("Timeout waiting for reset response.")
                observations.append(None)
        return np.array(observations)
    
    
    def step(self, actions):
        """Send actions to the subprocesses and collect the results."""
        for i, (q_in, _) in enumerate(self.queues):
            q_in.put(('step', actions[i]))

        results = []
        for _, q_out in self.queues:
            result = q_out.get()
            results.append(result)
        obs, rewards, dones, truncateds, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), np.array(truncateds), infos

    def close(self):
        """Close the environments."""
        if not self.closed:
            for q_in, _ in self.queues:
                q_in.put(('close', None))
            for p in self.processes:
                p.join()
            self.closed = True

    def env_is_wrapped(self, wrapper_class):
        """Return True if the environment is wrapped with a specific wrapper class."""
        return True

    def env_method(self, method_name, *args, **kwargs):
        """Call a method on all environments."""
        for q_in, _ in self.queues:
            q_in.put(('method', method_name, args, kwargs))

        results = [q_out.get() for _, q_out in self.queues]
        return results

    def get_attr(self, attr_name, index=0):
        """Get an attribute from all environments."""
        for q_in, _ in self.queues:
            q_in.put(('get_attr', attr_name, index))

        results = [q_out.get() for _, q_out in self.queues]
        return results

    def set_attr(self, attr_name, value, index=0):
        """Set an attribute on all environments."""
        for q_in, _ in self.queues:
            q_in.put(('set_attr', attr_name, value, index))

        results = [q_out.get() for _, q_out in self.queues]
        return results

    def step_async(self, actions):
        """Send actions to the environments."""
        for i, (q_in, _) in enumerate(self.queues):
            q_in.put(('step', actions[i]))

    def step_wait(self):
        """Wait for and collect the results of the actions."""
        results = [q_out.get() for _, q_out in self.queues]
        obs, rewards, dones, truncateds, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), np.array(truncateds), infos
    
    def get_num_envs(self):
        return self.num_envs
    
    @property
    def action_space(self):
        """Return the action space of the environment."""
        if self.envs[0] is not None:
            return self.envs[0].action_space
        else:
            raise AttributeError("Environment not initialized properly.")

    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        if self.envs[0] is not None:
            return self.envs[0].observation_space
        else:
            raise AttributeError("Environment not initialized properly.")
    