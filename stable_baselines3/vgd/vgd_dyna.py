from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
import math as _math
import wandb

try:
    from model.diffusion.sampling import make_timesteps as _make_timesteps
    from model.diffusion.sampling import extract as _extract
except Exception:
    _make_timesteps = None
    _extract = None


SelfVGD = TypeVar("SelfVGD", bound="VGD_DYNA")



class VGD_DYNA(OffPolicyAlgorithm):
    """
    Value-Guided Denoising (VGD): guidance-only decode with x0 gradients.

    This class runs environment interaction using a frozen diffusion policy guided by
    the environment critic. There is no learned noise actor or noise-critic. Guidance
    is always enabled (guidance_only=True) and uses x0 gradients.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        diffusion_policy=None,
        diffusion_act_dim=None,
        critic_backup_combine_type: str = 'min',
        target_uncertainty: float = 0.01,
        uncertainty_beta: float = 0.1,
        guidance_lambda: float = 0.0,
        guidance_warmup_steps: int = 0,
        guidance_time_anneal: str = 'linear',
        guidance_last_k_steps: int = 0,
        guidance_decode_steps: int = -1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_update_interval = target_update_interval
        self.diffusion_policy = diffusion_policy
        self.diffusion_act_chunk = diffusion_act_dim[0]
        self.diffusion_act_dim = diffusion_act_dim[1]
        self.critic_backup_combine_type = critic_backup_combine_type
        # Guided denoising parameters (x0 guidance only)
            
        self.target_uncertainty = target_uncertainty
        self.uncertainty_beta = uncertainty_beta
        self.guidance_lambda = guidance_lambda
        self.guidance_warmup_steps = int(guidance_warmup_steps)
        print(f"DEBUG: Guidance Warmup Steps is set to: {self.guidance_warmup_steps}")
        self.guidance_time_anneal = guidance_time_anneal
        self.guidance_last_k_steps = int(guidance_last_k_steps)
        self.guidance_decode_steps = int(guidance_decode_steps)
        # Accumulators for guidance metrics (averaged at flush time)
        self._guide_acc = {
            "grad_norm": 0.0,
            "weighted_grad_norm": 0.0,
            "delta_norm": 0.0,
            "ratio_weighted_grad_to_delta": 0.0,
            "cosine": 0.0,
            "cosine_pos_frac": 0.0,
            "q_pre": 0.0,
            "q_post": 0.0,
            "delta_q": 0.0,
            "lambda_mean": 0.0,
            "lambda_base": 0.0,
            "weight_mean": 0.0,
        }
        self._guide_count = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        optimizers = [self.critic.optimizer]
        self._update_learning_rate(optimizers)

        critic_losses = []
        # [추가됨] Q값 로깅을 위한 리스트 초기화
        n_critics = self.critic.n_critics
        q_means_history = [[] for _ in range(n_critics)]
        td_errors_batch = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Next actions via guided decode of random noise
                next_noise = th.randn(
                    replay_data.next_observations.shape[0],
                    self.diffusion_act_chunk,
                    self.diffusion_act_dim,
                    device=self.device,
                )
            
                next_actions = self._guided_decode(replay_data.next_observations, next_noise, deterministic=True)
                next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)

                # Compute the next Q values and target
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                if self.critic_backup_combine_type == 'min':
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                else:
                    next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                min_possible_return = -1.0 * 400  # Reward(-1) * Max_Steps(400)
                target_q_values = th.clamp(target_q_values, min=min_possible_return, max=400.0)
            # Current Q-values for actions stored in buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            
            for current_q in current_q_values:
                td_error_per_sample = th.abs(current_q - target_q_values)
                td_errors_batch.append(td_error_per_sample)
            
            
            for i in range(n_critics):
                q_means_history[i].append(current_q_values[i].mean().item())
            # Critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        
        
        all_td_errors = th.cat(td_errors_batch, dim=0)
        self._recent_td_error_mean = all_td_errors.mean().item()
        self._recent_td_error_std = all_td_errors.std().item()

    
        self._flush_guidance_logs()
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        q_vals_all_means = []
        for i in range(n_critics):
            mean_val = np.mean(q_means_history[i])
            self.logger.record(f"train/q{i+1}_mean", mean_val)
            q_vals_all_means.append(mean_val)
        
        if n_critics > 1:
            q_std = np.std(q_vals_all_means)
            self.logger.record("train/q_std_inter_critic", q_std)
            
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        # random warmup is irrelevant as we ignore actor; keep for compatibility
        if self.num_timesteps < learning_starts:
            pass
        assert self._last_obs is not None, "self._last_obs was not set"
        obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
        init_noise = th.tanh(th.randn(obs.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim, device=self.device))
        decoded = self._guided_decode(obs, init_noise, deterministic=True)
        action = decoded.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
        action = action.detach().cpu().numpy()
        buffer_action = action
        return action, buffer_action

    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        init_noise = th.tanh(th.randn(obs.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim, device=self.device))
        decoded = self._guided_decode(obs, init_noise, deterministic=True)
        action = decoded.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
        action = action.detach().cpu().numpy()
        return action, None

    def predict_noise_latent(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> np.ndarray:
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        noise = th.tanh(th.randn(obs.shape[0], self.diffusion_act_chunk, self.diffusion_act_dim, device=self.device))
        return noise.detach().cpu().numpy()

    def _current_lambda(self) -> float:
        if self.guidance_lambda <= 0.0:
            return 0.0
        if self.guidance_warmup_steps <= 0:
            return float(self.guidance_lambda)
        progress = min(1.0, float(self._n_updates) / float(self.guidance_warmup_steps))
        return float(self.guidance_lambda) * progress

    def _time_anneal(self, step_index: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 1.0
        t = (step_index + 1) / float(total_steps)
        if self.guidance_time_anneal == 'off':
            return 1.0
        if self.guidance_time_anneal == 'cosine':
            return 0.5 - 0.5 * _math.cos(_math.pi * t)
        return t

    def _guided_decode(self, obs: th.Tensor, initial_noise: th.Tensor, deterministic: bool = True) -> th.Tensor:
        base_wrapper = self.diffusion_policy
        base_policy = getattr(base_wrapper, 'base_policy', None)
        if base_policy is None:
            return base_wrapper(obs, initial_noise, return_numpy=False)
        device = base_policy.betas.device if hasattr(base_policy, 'betas') else self.device
        B = initial_noise.shape[0]
        x = initial_noise.to(device)
        sum_grad_norm = 0.0
        sum_weighted_grad_norm = 0.0
        sum_delta_norm = 0.0
        sum_ratio = 0.0
        sum_cos = 0.0
        count_cos_pos = 0.0
        sum_q_pre = 0.0
        sum_q_post = 0.0
        sum_lambda_t = 0.0
        sum_weight = 0.0
        # Determine schedule
        if getattr(base_policy, 'use_ddim', False):
            t_all = base_policy.ddim_t
        else:
            t_all = list(reversed(range(base_policy.denoising_steps)))
        total_steps = len(t_all)
        for i, t in enumerate(t_all):
            t_b = _make_timesteps(B, t, device) if _make_timesteps is not None else th.full((B,), t, device=device, dtype=th.long)
            index_b = _make_timesteps(B, i, device) if _make_timesteps is not None else th.full((B,), i, device=device, dtype=th.long)
            x_prev = x
            mean, logvar = base_policy.p_mean_var(x=x, t=t_b, cond={"state": obs.to(device), "noise_action": x}, index=index_b, deterministic=deterministic)
            std = th.exp(0.5 * logvar)
            if getattr(base_policy, 'use_ddim', False):
                std = th.zeros_like(std)
            else:
                if t == 0:
                    std = th.zeros_like(std)
                else:
                    std = th.clamp(std, min=1e-3)
            noise = th.zeros_like(x) if getattr(base_policy, 'use_ddim', False) else th.randn_like(x).clamp_(
                -getattr(base_policy, 'randn_clip_value', 10), getattr(base_policy, 'randn_clip_value', 10)
            )
            x_model = mean + std * noise
            delta = (x_model - x_prev).reshape(B, -1)
            x = x_model
            lambda_base = self._current_lambda()
            apply_guidance = lambda_base > 0.0 and (self.guidance_last_k_steps <= 0 or i >= total_steps - self.guidance_last_k_steps)
            if apply_guidance:
                # Ensure gradients are enabled even if caller wrapped in no_grad
                with th.enable_grad():
                    # x0 guidance: deterministically finish to x0_hat, compute dQ/dx0, nudge x0, recompute mu
                    with th.no_grad():
                        z_det = x
                        steps_remain = total_steps - i - 1
                        for j in range(i + 1, i + 1 + steps_remain):
                            t_dec = t_all[j]
                            tb_dec = _make_timesteps(B, t_dec, device) if _make_timesteps is not None else th.full((B,), t_dec, device=device, dtype=th.long)
                            idx_dec = _make_timesteps(B, j, device) if _make_timesteps is not None else th.full((B,), j, device=device, dtype=th.long)
                            mu_dec, _ = base_policy.p_mean_var(x=z_det, t=tb_dec, cond={"state": obs.to(device), "noise_action": z_det}, index=idx_dec, deterministic=True)
                            z_det = mu_dec
                    z_var = z_det.detach().requires_grad_(True)
                    a_flat = z_var.reshape(B, -1)
                    q_vals_all = th.cat(self.critic(obs.to(self.device), a_flat.to(self.device)), dim=1)
                    if self.critic_backup_combine_type == 'min':
                        q_vals, _ = th.min(q_vals_all, dim=1, keepdim=True)
                    else:
                        q_vals = th.mean(q_vals_all, dim=1, keepdim=True)
                    grad_z = th.autograd.grad(q_vals.sum(), z_var, retain_graph=False, create_graph=False)[0]
                    lam_t = float(lambda_base * self._time_anneal(i, total_steps))

      
                    if self.num_timesteps < self.guidance_warmup_steps:
                        # Original Logic: x0 = z + lambda * grad
                        x0_guided = z_var + lam_t * grad_z.detach()
                        
                        step_size = th.full((B, 1), lam_t, device=device)
                        current_weight_val = 1.0

                    else:
                        if hasattr(self, '_recent_td_error_mean'):
                            td_normalized = self._recent_td_error_mean / 100.0
                            global_weight = np.exp(-self.uncertainty_beta * td_normalized)
                        else:
                            global_weight = 1.0
                            
                        # Local component (per-sample)
                        q_std = q_vals_all.std(dim=1, keepdim=True)
                        std_normalized = q_std / 5.0
                        local_weight = th.exp(-self.uncertainty_beta * std_normalized).squeeze(-1)
                        
                        final_weight = local_weight * global_weight
                        
                        # 최소값 보장
                        uncertainty_weight = th.clamp(final_weight, min=0.001, max=1.0)

                    
                        if z_var.ndim == 3:
                            weight_reshaped = uncertainty_weight.view(B, 1, 1)
                        else:
                            weight_reshaped = uncertainty_weight.view(B, 1)
                            
                        # Apply
                        final_step_size = lam_t * weight_reshaped
                        x0_guided = z_var + final_step_size * grad_z.detach()
                        
                
                        step_size = final_step_size.reshape(B, -1).mean(dim=1, keepdim=True)
                        current_weight_val = uncertainty_weight.mean().item()
                    # ============================================================
                    # Action Clamping 
                    # ============================================================

                    clip_val = getattr(base_policy, "final_action_clip_value", 1.0)
                    if clip_val is None: clip_val = 1.0
                    x0_guided = th.clamp(x0_guided, -clip_val, clip_val)
                        
                    
                    # Only implement for DDIM path
                    if getattr(base_policy, 'use_ddim', False) and _extract is not None:
                        cond = {"state": obs.to(device), "noise_action": x}
                        pred_noise = base_policy.network(x, t_b, cond=cond)
                        alpha = _extract(base_policy.ddim_alphas, index_b, x.shape)
                        alpha_prev = _extract(base_policy.ddim_alphas_prev, index_b, x.shape)
                        sigma = _extract(base_policy.ddim_sigmas, index_b, x.shape)
                        dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * pred_noise
                        mu_guided = (alpha_prev**0.5) * x0_guided + dir_xt
                        x = mu_guided
                        grad = grad_z
                    else:
                        raise RuntimeError("Schedules missing for x0 guidance. Ensure DDIM schedules and _extract are available.")
                # stats
                scale_per_sample = step_size.reshape(B, -1).mean(dim=1) 

                grad_flat = grad.reshape(B, -1)
                delta_flat = delta.reshape(B, -1)

                grad_norm_vec = th.linalg.vector_norm(grad_flat, dim=1)
                delta_norm_vec = th.linalg.vector_norm(delta_flat, dim=1)

                weighted_grad_norm_vec = scale_per_sample * grad_norm_vec

                ratio_vec = weighted_grad_norm_vec / (delta_norm_vec + 1e-8)

                cos_vec = F.cosine_similarity(grad_flat, delta, dim=1, eps=1e-8)

                grad_norm = grad_norm_vec.mean().item()
                weighted_grad_norm = weighted_grad_norm_vec.mean().item() 
                delta_norm = delta_norm_vec.mean().item()
                
                ratio_mean = ratio_vec.mean().item()

                cos_mean = cos_vec.mean().item()
                cos_pos = (cos_vec > 0).float().mean().item()
                

                with th.no_grad():
                    a_post = x0_guided.reshape(B, -1)
                    q_post_all = th.cat(self.critic(obs.to(self.device), a_post.to(self.device)), dim=1)
                    if self.critic_backup_combine_type == 'min':
                        q_post, _ = th.min(q_post_all, dim=1, keepdim=True)
                    else:
                        q_post = th.mean(q_post_all, dim=1, keepdim=True)
                    q_post_mean = q_post.mean().item()
                    
                    a_det = z_det.reshape(B, -1)
                    q_pre_all = th.cat(self.critic(obs.to(self.device), a_det.to(self.device)), dim=1)
                    if self.critic_backup_combine_type == 'min':
                        q_pre, _ = th.min(q_pre_all, dim=1, keepdim=True)
                    else:
                        q_pre = th.mean(q_pre_all, dim=1, keepdim=True)
                    q_pre_mean = q_pre.mean().item()

                sum_grad_norm += grad_norm
                sum_weighted_grad_norm += weighted_grad_norm
                sum_delta_norm += delta_norm
                sum_ratio += ratio_mean  
                sum_cos += cos_mean
                count_cos_pos += cos_pos
                sum_q_pre += q_pre_mean
                sum_q_post += q_post_mean
                sum_weight += current_weight_val
                sum_lambda_t += scale_per_sample.mean().item() 

            if getattr(base_policy, 'final_action_clip_value', None) is not None and i == total_steps - 1:
                x = th.clamp(x, -base_policy.final_action_clip_value, base_policy.final_action_clip_value)
        if total_steps > 0 and self._current_lambda() > 0.0:
            if self.guidance_last_k_steps > 0:
                steps = float(min(self.guidance_last_k_steps, total_steps))
            else:
                steps = float(total_steps)
            vals = {
                "grad_norm": sum_grad_norm / steps,
                "weighted_grad_norm": sum_weighted_grad_norm / steps,
                "delta_norm": sum_delta_norm / steps,
                "ratio_weighted_grad_to_delta": sum_ratio / steps,
                "cosine": sum_cos / steps,
                "cosine_pos_frac": count_cos_pos / steps,
                "q_pre": sum_q_pre / steps,
                "q_post": sum_q_post / steps,
                "delta_q": (sum_q_post - sum_q_pre) / steps,
                "lambda_mean": sum_lambda_t / steps,
                "lambda_base": self._current_lambda(),
                "weight_mean": sum_weight / steps 
            }
            for k, v in vals.items():
                self._guide_acc[k] += float(v)
            self._guide_count += 1
        return x

    def _flush_guidance_logs(self) -> None:
        if self._guide_count <= 0:
            return
        avg = {k: (self._guide_acc[k] / float(self._guide_count)) for k in self._guide_acc}
        for k, v in avg.items():
            self.logger.record(f"guide/{k}", v)
        # try:
        #     if getattr(wandb, "run", None) is not None:
        #         wandb.log({**{f"guide/{k}": v for k, v in avg.items()}}, commit=False)
        # except Exception:
        #     pass
        for k in self._guide_acc:
            self._guide_acc[k] = 0.0
        self._guide_count = 0


