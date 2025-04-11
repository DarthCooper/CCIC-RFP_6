# Native imports
from typing import Tuple


class BuildFroggerConfig:
    num_envs : int = 4
    env_name : str = "ALE/Frogger-v5"
    render_mode : str = "rgb_array"
    max_episode_steps : int = 50000
    disable_env_checker : bool = None
    full_action_space : bool = False

class PreprocessingFroggerConfig:
    noop_max : int = 30
    frame_skip : int = 1
    screen_size: int | Tuple[int, int] = (84, 84)
    terminal_on_life_loss : bool = False
    grayscale_obs : bool = True
    grayscale_newaxis : bool = True
    scale_obs : bool = False

class MemoryFroggerConfig:
    capacity : int = 100000
    save_freq : int = 5
    save_dir : str = "./checkpoints"
    model_path : str = "dqn_model.keras"
    memory_path : str = "training_memory.pkl"

class ModelConfig:
    episodes : int = 10000
    batch_size : int = 32
    gamma : float = 0.99
    epsilon : float = 1.0
    epsilon_min : float = 0.1
    epsilon_decay : float = 0.995
    update_target_freq : int = 10000