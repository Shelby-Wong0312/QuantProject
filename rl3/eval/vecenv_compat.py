from typing import Callable, Sequence
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

class _CompatExposeSingleSpace(VecEnvWrapper):
    """
    薄相容層：
    - 在 __init__ 就把 observation_space / action_space 傳給父類，避免 BaseVecEnv 再寫入
    - 提供 single_action_space / single_observation_space / num_envs
    - 完整轉發必要介面：reset / step_async / step_wait / close
    """
    def __init__(self, venv):
        # 將 space 直接帶入，避免 BaseVecEnv.__init__ 對屬性賦值衝突
        super().__init__(venv, observation_space=venv.observation_space, action_space=venv.action_space)

        # 透傳單環境 space（若沒有 single_*，就用一般 space 退而求其次）
        self.single_action_space = getattr(venv, "single_action_space", getattr(venv, "action_space", None))
        self.single_observation_space = getattr(venv, "single_observation_space", getattr(venv, "observation_space", None))

        # 透傳 num_envs
        self.num_envs = getattr(venv, "num_envs", 1)

    # === 必要方法轉發 ===
    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def close(self):
        return self.venv.close()

def make_sb3_vecenv(make_env_fns: Sequence[Callable[[], gym.Env]], use_subproc: bool = False):
    """
    統一使用 SB3 的 VecEnv，並套相容層以暴露 single_* 屬性。
    """
    if use_subproc and len(make_env_fns) > 1:
        venv = SubprocVecEnv(make_env_fns)
    else:
        venv = DummyVecEnv(make_env_fns)

    venv = _CompatExposeSingleSpace(venv)

    # 基本驗收（避免無效 shape）
    assert getattr(venv, "single_action_space", None) is not None, "VecEnv must expose single_action_space."
    shape = getattr(getattr(venv, "single_action_space", None), "shape", None)
    assert shape is not None and len(shape) > 0 and all(s > 0 for s in shape), f"Invalid single_action_space.shape: {shape}"

    return venv
