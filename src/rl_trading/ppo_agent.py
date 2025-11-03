"""
PPO (Proximal Policy Optimization) Agent
PPO 強化學習算法實現 - 日內交易智能體
Cloud Quant - Task DT-002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO 配置參數"""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    n_steps: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    """
    Actor-Critic 網絡

    同時輸出動作概率（Actor）和狀態價值（Critic）
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 3
    ):
        """
        初始化 Actor-Critic 網絡

        Args:
            obs_dim: 觀察空間維度
            action_dim: 動作空間維度
            hidden_dim: 隱藏層大小
            n_layers: 網絡層數
        """
        super(ActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 共享特徵提取層
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Actor 網絡（政策）
        actor_layers = []
        for i in range(n_layers - 1):
            actor_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
        actor_layers.append(nn.Linear(hidden_dim, action_dim))

        self.actor = nn.Sequential(*actor_layers)

        # Critic 網絡（價值函數）
        critic_layers = []
        for i in range(n_layers - 1):
            critic_layers.extend(
                [
                    nn.Linear(
                        hidden_dim, hidden_dim // 2 if i == n_layers - 2 else hidden_dim
                    ),
                    nn.LayerNorm(hidden_dim // 2 if i == n_layers - 2 else hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
        critic_layers.append(nn.Linear(hidden_dim // 2, 1))

        self.critic = nn.Sequential(*critic_layers)

        # 初始化權重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化網絡權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            obs: 觀察張量

        Returns:
            action_logits: 動作邏輯值
            value: 狀態價值
        """
        # 展平觀察（如果是多維的）
        if len(obs.shape) > 2:
            batch_size = obs.shape[0]
            obs.view(batch_size, -1)

        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        獲取動作和價值

        Args:
            obs: 觀察
            action: 指定的動作（用於計算 log_prob）

        Returns:
            action, log_prob, entropy, value
        """
        action_logits, value = self.forward(obs)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        獲取狀態價值

        Args:
            obs: 觀察

        Returns:
            狀態價值
        """
        _, value = self.forward(obs)
        return value.squeeze(-1)


class RolloutBuffer:
    """
    經驗回放緩衝區

    存儲訓練數據並計算優勢函數
    """

    def __init__(self, buffer_size: int, obs_shape: Tuple, device: str = "cpu"):
        """
        初始化緩衝區

        Args:
            buffer_size: 緩衝區大小
            obs_shape: 觀察形狀
            device: 設備
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device

        # 預分配內存
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # GAE 計算
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size

    def add(self, obs, action, reward, value, log_prob, done):
        """添加經驗"""
        assert self.ptr < self.max_size

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr += 1

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ):
        """
        計算回報和優勢函數 (GAE)

        Args:
            last_value: 最後狀態的價值
            gamma: 折扣因子
            gae_lambda: GAE lambda
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        # 添加最後的價值
        values = np.append(values, last_value)

        # GAE 計算
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]

            delta = (
                rewards[step] + gamma * next_value * next_non_terminal - values[step]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[self.path_start_idx + step] = gae
            self.returns[self.path_start_idx + step] = gae + values[step]

    def get(self) -> Dict[str, torch.Tensor]:
        """獲取所有數據"""
        assert self.ptr == self.max_size

        # 標準化優勢
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

        dict(
            observations=self.observations,
            actions=self.actions,
            values=self.values,
            log_probs=self.log_probs,
            advantages=self.advantages,
            returns=self.returns,
        )

        return {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in data.items()
        }

    def reset(self):
        """重置緩衝區"""
        self.ptr = 0
        self.path_start_idx = 0


class PPOTrainer:
    """
    PPO 訓練器

    管理訓練循環和優化過程
    """

    def __init__(self, env, config: PPOConfig = None):
        """
        初始化 PPO 訓練器

        Args:
            env: 交易環境
            config: PPO 配置
        """
        self.env = env
        self.config = config or PPOConfig()

        # 獲取環境信息
        obs_shape = env.observation_space.shape
        obs_dim = np.prod(obs_shape)
        action_dim = env.action_space.n

        # 創建模型
        self.model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(
            self.config.device
        )

        # 優化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

        # 緩衝區
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            obs_shape=obs_shape,
            device=self.config.device,
        )

        # 訓練統計
        self.total_timesteps = 0
        self.n_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # 性能指標
        self.best_reward = -float("inf")
        self.training_history = []

        logger.info(f"PPO Trainer initialized on {self.config.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

    def collect_rollouts(self) -> bool:
        """
        收集訓練數據

        Returns:
            是否收集成功
        """
        n_steps = 0
        self.rollout_buffer.reset()

        # 重置環境（如果需要）
        if not hasattr(self, "obs"):
            self.obs, _ = self.env.reset()
            self.episode_reward = 0
            self.episode_length = 0

        # 收集經驗
        while n_steps < self.config.n_steps:
            with torch.no_grad():
                # 轉換觀察到張量
                obs_tensor = torch.tensor(
                    self.obs, dtype=torch.float32, device=self.config.device
                )
                obs_tensor = obs_tensor.unsqueeze(0)  # 添加 batch 維度

                # 獲取動作和價值
                action, log_prob, _, value = self.model.get_action_and_value(obs_tensor)

                action = action.cpu().numpy()[0]
                value = value.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

            # 執行動作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 記錄
            self.episode_reward += reward
            self.episode_length += 1
            self.total_timesteps += 1

            # 添加到緩衝區
            self.rollout_buffer.add(self.obs, action, reward, value, log_prob, done)

            self.obs = next_obs
            n_steps += 1

            # Episode 結束
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_lengths.append(self.episode_length)
                self.n_episodes += 1

                # 記錄最佳
                if self.episode_reward > self.best_reward:
                    self.best_reward = self.episode_reward

                # 重置
                self.obs, _ = self.env.reset()
                self.episode_reward = 0
                self.episode_length = 0

        # 計算最後狀態的價值
        with torch.no_grad():
            obs_tensor = torch.tensor(
                self.obs, dtype=torch.float32, device=self.config.device
            )
            obs_tensor = obs_tensor.unsqueeze(0)
            last_value = self.model.get_value(obs_tensor).cpu().numpy()[0]

        # 計算 GAE
        self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        return True

    def train_step(self) -> Dict[str, float]:
        """
        執行一次訓練步驟

        Returns:
            訓練指標
        """
        # 獲取數據
        rollout_data = self.rollout_buffer.get()

        # 訓練統計
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []

        # 多輪訓練
        for epoch in range(self.config.n_epochs):
            # 生成批次索引
            n_samples = self.config.n_steps
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, self.config.batch_size):
                batch_indices = indices[start_idx : start_idx + self.config.batch_size]

                # 批次數據
                batch = {k: v[batch_indices] for k, v in rollout_data.items()}

                # 獲取當前策略的動作概率和價值
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch["observations"], batch["actions"].long()
                )

                # 計算比率
                ratio = torch.exp(new_log_probs - batch["log_probs"])

                # Clipped surrogate loss
                advantages = batch["advantages"]
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                    )
                    * advantages
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch["returns"])

                # Entropy loss
                entropy_loss = -entropy.mean()

                # 總損失
                loss = (
                    pg_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # 優化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # 記錄
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                # Clip fraction
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > self.config.clip_ratio).float()
                )
                clip_fractions.append(clip_fraction.item())

        # 更新學習率
        self.scheduler.step()

        # 返回統計
        return {
            "pg_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def train(
        self, total_timesteps: int, log_interval: int = 10, save_interval: int = 100
    ):
        """
        訓練主循環

        Args:
            total_timesteps: 總訓練步數
            log_interval: 日誌間隔
            save_interval: 保存間隔
        """
        iteration = 0

        while self.total_timesteps < total_timesteps:
            # 收集數據
            self.collect_rollouts()

            # 訓練
            train_stats = self.train_step()

            iteration += 1

            # 記錄
            if iteration % log_interval == 0:
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_length = np.mean(self.episode_lengths)

                    logger.info(f"Iteration {iteration}")
                    logger.info(f"  Timesteps: {self.total_timesteps}")
                    logger.info(f"  Episodes: {self.n_episodes}")
                    logger.info(f"  Mean reward: {mean_reward:.2f}")
                    logger.info(f"  Mean length: {mean_length:.0f}")
                    logger.info(f"  Best reward: {self.best_reward:.2f}")
                    logger.info(f"  PG loss: {train_stats['pg_loss']:.4f}")
                    logger.info(f"  Value loss: {train_stats['value_loss']:.4f}")
                    logger.info(f"  Clip fraction: {train_stats['clip_fraction']:.3f}")

                    self.training_history.append(
                        {
                            "iteration": iteration,
                            "timesteps": self.total_timesteps,
                            "mean_reward": mean_reward,
                            "best_reward": self.best_reward,
                            **train_stats,
                        }
                    )

            # 保存模型
            if iteration % save_interval == 0:
                self.save_model(f"ppo_trader_iter_{iteration}.pt")

    def save_model(self, path: str):
        """保存模型"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "total_timesteps": self.total_timesteps,
                "n_episodes": self.n_episodes,
                "best_reward": self.best_reward,
                "training_history": self.training_history,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.total_timesteps = checkpoint["total_timesteps"]
        self.n_episodes = checkpoint["n_episodes"]
        self.best_reward = checkpoint["best_reward"]
        self.training_history = checkpoint["training_history"]
        logger.info(f"Model loaded from {path}")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        評估模型

        Args:
            n_episodes: 評估回合數

        Returns:
            評估指標
        """
        self.model.eval()

        episode_rewards = []
        episode_lengths = []
        win_rates = []
        max_drawdowns = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=self.config.device
                    )
                    obs_tensor = obs_tensor.unsqueeze(0)
                    action, _, _, _ = self.model.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 從環境信息獲取更多指標
            if "win_rate" in info:
                win_rates.append(info["win_rate"])
            if "max_drawdown" in info:
                max_drawdowns.append(info["max_drawdown"])

        self.model.train()

        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "total_return": np.sum(episode_rewards),
        }

        if win_rates:
            results["mean_win_rate"] = np.mean(win_rates)
        if max_drawdowns:
            results["mean_max_drawdown"] = np.mean(max_drawdowns)

        return results


if __name__ == "__main__":
    print("PPO Agent Implementation - Cloud Quant Task DT-002")
    print("=" * 50)
    print("Core components:")
    print("- ActorCritic network with LayerNorm and Dropout")
    print("- Generalized Advantage Estimation (GAE)")
    print("- Clipped surrogate objective")
    print("- Parallel rollout collection")
    print("- Automatic model checkpointing")
    print("\nReady for training!")
