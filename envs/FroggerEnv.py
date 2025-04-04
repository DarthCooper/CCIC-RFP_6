import gymnasium as gym
import numpy as np
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from stable_baselines3.common.vec_env import VecEnv
from gymnasium.wrappers import AtariPreprocessing
from queue import Empty

import ale_py

def worker(idx, q_in, q_out, env):
    """Worker process that creates the environment and handles requests."""
    print(f"Worker {idx} process started.")  # Confirm worker is starting
    while True:
        try:
            command, data = q_in.get(timeout=1)  # Wait for a command
            print(f"Worker {idx} received command: {command}")  # Debug command received
            if command == 'reset':
                print(f"Worker {idx} resetting environment...")
                obs, info = env.reset()  # Reset the environment and get observation
                q_out.put((obs, info))
            elif command == 'step':
                print(f"Worker {idx} stepping environment...")
                obs, reward, done, truncated, info = env.step(data)
                q_out.put((obs, reward, done, truncated, info))
            elif command == 'close':
                print(f"Worker {idx} closing environment...")
                env.close()
                break
        except Empty:
            continue

class CustomVecEnv(VecEnv):
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [None] * num_envs  # Initialize list of environments
        self.processes = []
        self.queues = []
        self.closed = False

        # Use multiprocessing.Manager to manage shared state for environments
        manager = multiprocessing.Manager()
        self.envs = manager.list([None] * num_envs)  # Shared list to hold environments

        # Create the environment processes
        for i in range(self.num_envs):
            env = gym.make("ALE/Frogger-v5", render_mode="rgb_array",)
            env = AtariPreprocessing(env, frame_skip=1, grayscale_newaxis=True)

            q_in = multiprocessing.Queue()
            q_out = multiprocessing.Queue()
            self.queues.append((q_in, q_out))
            # Create a separate process for each environment worker
            p = multiprocessing.Process(target=worker, args=(i, q_in, q_out, env))  # Pass index of env and shared envs list
            print(f"Main process starting worker {i}")  # Debug message
            p.start()
            self.processes.append(p)
            self.envs[i] = env

    def reset(self):
        """Reset all environments and return observations."""
        print("Main process sending reset command...", flush=True)
        for q_in, _ in self.queues:
            q_in.put(('reset', None))  # Send reset command to each worker

        observations = []
        for _, q_out in self.queues:
            try:
                print("Main process waiting for response...", flush=True)
                obs, info = q_out.get(timeout=5)  # Wait for response with timeout to avoid hanging
                print(f"Main process received observation with shape: {obs.shape}", flush=True)
                observations.append(obs)
            except Empty:
                print("Timeout waiting for reset response.")
                observations.append(None)
        return np.array(observations)

    def step(self, action):
        """Send actions to the subprocesses and collect the results."""
        for i, (q_in, _) in enumerate(self.queues):
            q_in.put(('step', action))
            print(f"action {action} performed on index {i}")

        # Collect the results from all environments
        results = []
        for _, q_out in self.queues:
            result = q_out.get()
            results.append(result)
            print(f"Step result from environment: {result}")  # Print step result from each environment
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

    # These methods are required by VecEnv
    def env_is_wrapped(self, wrapper_class):
        """Return True if the environment is wrapped with a specific wrapper class."""
        return False  # Change this if you are using wrappers

    def env_method(self, method_name, *args, **kwargs):
        """Call a method on all environments."""
        for q_in, _ in self.queues:
            q_in.put(('method', method_name, args, kwargs))

        # Collect results
        results = [q_out.get() for _, q_out in self.queues]
        return results

    def get_attr(self, attr_name, index=0):
        """Get an attribute from all environments."""
        for q_in, _ in self.queues:
            q_in.put(('get_attr', attr_name, index))

        # Collect results
        results = [q_out.get() for _, q_out in self.queues]
        return results

    def set_attr(self, attr_name, value, index=0):
        """Set an attribute on all environments."""
        for q_in, _ in self.queues:
            q_in.put(('set_attr', attr_name, value, index))

        # Collect results
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
    
    @property
    def action_space(self):
        """Return the action space of the environment."""
        if self.envs[0] is not None:
            return self.envs[0].action_space  # Access action_space of first env
        else:
            raise AttributeError("Environment not initialized properly.")

    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        if self.envs[0] is not None:
            return self.envs[0].observation_space  # Access observation_space of first env
        else:
            raise AttributeError("Environment not initialized properly.")
    