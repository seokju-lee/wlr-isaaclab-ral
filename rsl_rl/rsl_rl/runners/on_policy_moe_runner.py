# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixture-of-Experts PPO runner with expert pretraining and MoE fine-tuning."""

from __future__ import annotations
import os
import sys
import time
import statistics
import torch
from collections import deque

# --- RSL-RL core imports ---
from rsl_rl.algorithms import PPO, Distillation, MOEPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state

# --- MoE policy ---
from rsl_rl.modules.actor_critic_moe import ActorCriticMoE


class OnPolicyMoERunner:
    """On-policy runner for Mixture-of-Experts (MoE) training with logging, saving and normalization."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        print(f"[DEBUG] Initializing LOCAL OnPolicyMoERunner from {__file__}")
        self.cfg = train_cfg
        # Stage and expert type provided in train_cfg control pretraining and merging behavior.
        self.stage = self.cfg.get("stage", None)
        self.expert_type = self.cfg.get("expert_type", None)
        self.expert_data_dir = self.cfg.get("expert_data_dir", None)
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Multi-GPU setup
        self._configure_multi_gpu()

        # Resolve training type
        if self.alg_cfg["class_name"] in ["PPO", "MOEPPO"]:
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # Resolve obs dims
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        # privileged obs: critic for RL, teacher for distillation (fallback to obs)
        if self.training_type == "rl":
            self.privileged_obs_type = "critic" if "critic" in extras["observations"] else None
        else:
            self.privileged_obs_type = "teacher" if "teacher" in extras["observations"] else None
        num_privileged_obs = extras["observations"].get(self.privileged_obs_type, obs).shape[1] if self.privileged_obs_type else num_obs

        # Build policy/algorithm
        policy_class = eval(self.policy_cfg.pop("class_name"))
        self.policy = policy_class(num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # MOEPPO | PPO | Distillation
        self.alg = alg_class(self.policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Storage/normalization config
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # RND cfg: if present, add num_states & dt scaling (same as OnPolicyRunner)
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            self.alg_cfg["rnd_cfg"]["num_states"] = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt  # scale with dt

        # Symmetry cfg: pass env for augmentation helper
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # Init storage for current algorithm
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Logging state
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.git_status_repos = []  # fill with repo roots for code-state snapshot if needed

        # Expert-specific settings
        # Expert-specific settings
        self.pretrain_iters = 2000  # Default, will be overridden in learn()
        self.valid_experts = ["straight", "turn", "drift"]

    # =====================================================================
    # Stage A: Expert pretraining
    # =====================================================================
    def pretrain_experts(self, env, expert_name: str):
        """Pretrain a single expert using the environment's current path configuration.

        The function assumes the environment's path sampling configuration (PATH_CFG)
        has been set prior to environment creation (for example by the training script).
        Training proceeds with a standard PPO actor/critic and saves checkpoints under
        <log_dir>/moe_checkpoints/expert_{expert_name}.
        """

        # Initialize writer if needed (same as runner.learn)
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = getattr(self, "logger_type", self.cfg.get("logger", "tensorboard")).lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Use current env path configuration and train a single expert.
        # Resolve obs dimensions
        obs, extras = env.get_observations()
        num_obs = obs.shape[1]
        critic_obs = extras["observations"].get("critic", obs)
        num_privileged_obs = critic_obs.shape[1]

        # Build a fresh PPO agent for this expert
        expert_policy = ActorCritic(
            num_obs, num_privileged_obs, env.num_actions, init_noise_std=1.0,
            actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128],
            activation="elu"
        ).to(self.device)
        expert_alg = PPO(expert_policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Init storage for this expert
        expert_alg.init_storage(
            "rl", env.num_envs, self.num_steps_per_env,
            [num_obs], [num_privileged_obs], [env.num_actions],
        )

        # Logging buffers
        from collections import deque
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=self.device)

        # Initial observations (tensors on device)
        obs, extras = env.get_observations()
        obs = obs.to(self.device)
        critic_obs = extras["observations"].get("critic", obs).to(self.device)

        # Training loop for this expert
        start_iter = 0
        tot_iter = self.pretrain_iters
        num_learning_iterations = self.pretrain_iters

        SAVE_EVERY = 500
        for it in range(self.pretrain_iters):
            ep_infos = []
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    obs_n = self.obs_normalizer(obs)
                    critic_obs_n = self.privileged_obs_normalizer(critic_obs)

                    actions = expert_alg.act(obs_n, critic_obs_n)

                    next_obs, rew, dones, infos = env.step(actions.to(env.device))
                    next_obs, rew, dones = next_obs.to(self.device), rew.to(self.device), dones.to(self.device)

                    expert_alg.process_env_step(rew, dones, infos)

                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])

                    cur_reward_sum += rew
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    obs = next_obs
                    critic_obs = infos["observations"].get("critic", obs).to(self.device)

            collection_time = time.time() - start
            start = time.time()

            critic_obs_n = self.privileged_obs_normalizer(critic_obs).detach().clone()
            expert_alg.compute_returns(critic_obs_n)

            loss_dict = expert_alg.update()
            learn_time = time.time() - start

            if self.log_dir is not None and not self.disable_logs and self.writer is not None:
                _prev_alg = self.alg
                self.alg = expert_alg
                try:
                    self.current_learning_iteration = it
                    self.log({
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "ep_infos": ep_infos,
                        "loss_dict": loss_dict,
                        "it": it,
                        "tot_iter": tot_iter,
                        "start_iter": start_iter,
                        "num_learning_iterations": num_learning_iterations,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "erewbuffer": [],
                        "irewbuffer": [],
                    })
                finally:
                    self.alg = _prev_alg

            if it % 50 == 0:
                import statistics
                mr = statistics.mean(rewbuffer) if len(rewbuffer) else float("nan")
                el = statistics.mean(lenbuffer) if len(lenbuffer) else float("nan")
                print(f"[Expert {expert_name}] Iter {it}/{self.pretrain_iters} | mean_rew={mr:.3f} | ep_len={el:.1f} | loss={loss_dict}")

            if (it % SAVE_EVERY) == 0:
                # User requested: expert_name/time/policies
                # log_dir is now .../expert_name/time/
                ckpt_dir = self.log_dir
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"model_{it}.pt")

                # Save full checkpoint with normalizers
                saved_dict = {
                    "model_state_dict": expert_policy.state_dict(),
                    "optimizer_state_dict": expert_alg.optimizer.state_dict(),
                    "iter": it,
                    "infos": {},
                }
                if self.empirical_normalization:
                    saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
                    saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

                torch.save(saved_dict, ckpt_path)
                print(f"[Stage A] Expert {expert_name} checkpoint saved at: {ckpt_path}")

        final_dir = self.log_dir
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, f"model_{self.pretrain_iters}.pt")
        # Save both as model_X.pt and 'model_final.pt' for convencience? User asked for saving "policies" (plural).
        torch.save(expert_policy.state_dict(), final_path)
        print(f"[Stage A] Expert {expert_name} final saved at: {final_path}")

    # =====================================================================
    # Stage B: Merge Experts → MoE
    # =====================================================================

    def _load_expert_weights(self, moe_policy: ActorCriticMoE, expert_idx: int, loaded_dict: dict):
        """Helper to load weights from an expert checkpoint into the MoE policy at index expert_idx.
        Handles mapping from standard ActorCritic (actor.X -> actor_moe.experts[i].X)
        or single-expert MoE (actor_moe.experts.0.X -> actor_moe.experts[i].X).
        """
        if "model_state_dict" in loaded_dict:
            state_dict = loaded_dict["model_state_dict"]
        else:
            state_dict = loaded_dict

        new_state_dict = {}
        for k, v in state_dict.items():
            # 1) Standard ActorCritic -> MoE Expert i
            if k.startswith("actor."):
                suffix = k[len("actor."):]
                new_key = f"actor_moe.experts.{expert_idx}.{suffix}"
                new_state_dict[new_key] = v
            elif k.startswith("critic."):
                suffix = k[len("critic."):]
                new_key = f"critic_moe.experts.{expert_idx}.{suffix}"
                new_state_dict[new_key] = v
            # 2) Single-Expert MoE (index 0) -> MoE Expert i
            elif k.startswith("actor_moe.experts.0."):
                suffix = k[len("actor_moe.experts.0."):]
                new_key = f"actor_moe.experts.{expert_idx}.{suffix}"
                new_state_dict[new_key] = v
            elif k.startswith("critic_moe.experts.0."):
                suffix = k[len("critic_moe.experts.0."):]
                new_key = f"critic_moe.experts.{expert_idx}.{suffix}"
                new_state_dict[new_key] = v

        if not new_state_dict:
            print(f"[WARNING] No matching keys found in expert checkpoint for expert {expert_idx}! Keys in ckpt: {list(state_dict.keys())[:5]}...")
        else:
            # Load into the main policy
            # strict=False allows loading only the subset of keys corresponding to this expert
            moe_policy.load_state_dict(new_state_dict, strict=False)

            print(f"[INFO] Loaded {len(new_state_dict)} mapped keys for Expert {expert_idx}")

    def _derive_expert_arch(self, ckpt_path):
        """Analyze a checkpoint to infer hidden layer dimensions for actor and critic."""
        sd = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        def get_dims(prefix):
            # Check for standard actor.0.weight or actor_moe.experts.0.0.weight
            # We look for keys like "{prefix}.0.weight", "{prefix}.2.weight" etc.
            # OR "{prefix}_moe.experts.0.0.weight" if it was MoE.

            # Heuristic: Find keys starting with prefix, sort by number
            # Try standard keys first
            layers = {}  # index -> out_dim
            max_idx = -1

            # Map for keys: "actor.0.weight" -> 0, "actor.2.weight" -> 2
            # Also handle "actor_moe.experts.0.0.weight" -> 0

            candidates = [k for k in sd.keys() if (k.startswith(f"{prefix}.") or k.startswith(f"{prefix}_moe.experts.0."))]

            for k in candidates:
                if not k.endswith(".weight"):
                    continue
                parts = k.split(".")
                # Standard: actor.0.weight -> parts=["actor", "0", "weight"]
                # MoE: actor_moe.experts.0.0.weight -> parts=["actor_moe", "experts", "0", "0", "weight"]

                if parts[0] == prefix and parts[1].isdigit():
                    idx = int(parts[1])
                    out_dim = sd[k].shape[0]
                    layers[idx] = out_dim
                    if idx > max_idx:
                        max_idx = idx
                elif parts[0] == f"{prefix}_moe" and parts[3].isdigit():  # Experts layers
                    idx = int(parts[3])
                    out_dim = sd[k].shape[0]
                    layers[idx] = out_dim
                    if idx > max_idx:
                        max_idx = idx

            # Construct dims list
            # If we found layer 0 (out=256), layer 2 (out=128), layer 4 (out=12)
            # The *hidden* dims are [256, 128]. The last one is action/value dim.
            if max_idx == -1:
                return None

            hidden_dims = []
            # Iterate 0, 2, ...
            # Assuming interleaved activation (0, 1=act, 2, 3=act, ...)
            # We collect all except the last layer
            sorted_indices = sorted(layers.keys())
            for idx in sorted_indices[:-1]:
                hidden_dims.append(layers[idx])

            return hidden_dims

        actor_dims = get_dims("actor")
        critic_dims = get_dims("critic")
        return actor_dims, critic_dims

    # =====================================================================
    # Main learning pipeline
    # =====================================================================

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # Initialize writer (same behavior as OnPolicyRunner)
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Distillation readiness check
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # Randomize initial episode length
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # ===== Stage A: Pretrain Experts =====
        # Override pretrain_iters with the requested max_iterations
        if num_learning_iterations is not None:
            self.pretrain_iters = num_learning_iterations

        if self.stage == 1:
            if not self.expert_type:
                raise ValueError("Stage 1 requested but no expert_type provided. Provide --expert_type=straight|turn|drift")
            if self.expert_type not in self.valid_experts:
                raise ValueError(f"Requested expert '{self.expert_type}' not in valid experts: {self.valid_experts}")
            print(f"[INFO] Stage 1: Pretraining single expert: {self.expert_type}")
            self.pretrain_experts(self.env, self.expert_type)
            print("[INFO] Pretraining complete. Exiting as stage==1.")
            return

        # ===== Stage B: Merge Experts then Fine-tune MoE =====
        # Search strategy: {expert_data_dir}/{expert_name}/{latest_timestamp}/model_*.pt

        moe_expert_paths = []
        expected_experts = self.valid_experts

        expert_pool_dir = self.expert_data_dir

        # Auto-detect experiment root if expert_data_dir is not provided
        if expert_pool_dir is None:
            # Check if we are in a 'moe_merge' subfolder structure: .../experiment_name/moe_merge/timestamp/
            # self.log_dir = .../moe_merge/timestamp
            parent_dir = os.path.dirname(self.log_dir)  # .../moe_merge
            grandparent_dir = os.path.dirname(parent_dir)  # .../experiment_name

            if os.path.basename(parent_dir) == "moe_merge":
                print(f"[INFO] Auto-detected experiment root for expert search: {grandparent_dir}")
                expert_pool_dir = grandparent_dir
            else:
                # Fallback to log_dir (unlikely to work unless manually copied, but keeps old behavior)
                expert_pool_dir = self.log_dir

        if expert_pool_dir is None:
            raise ValueError("expert_data_dir (or valid log_dir) is required to locate pretrained experts for stage=2")

        missing_experts = []
        for name in expected_experts:
            # Look for directory: expert_pool_dir / name (e.g. expert_straight)
            # Or just 'straight' if arguments are passed that way? User arg is 'expert_type' but folder logic in train.py is exactly args.expert_type.
            # However valid_experts names are "straight", "turn", "drift".
            # so we look for expert_pool_dir/straight, etc.

            # Note: in train.py we set log_dir = .../expert_type/...
            # So if expert_type="straight", folder is .../straight/timestamp/...

            expert_type_dir = os.path.join(expert_pool_dir, name)

            # Also try "expert_{name}" just in case user named it that way?
            # But based on train.py logic, it uses valid_experts names directly unless user passed --expert_type="expert_straight"
            # Assuming standard "straight", "turn", "drift".

            if not os.path.isdir(expert_type_dir):
                # Try finding it? No, strict path.
                missing_experts.append(name)
                continue

            # Find latest timestamp folder
            subdirs = [d for d in os.listdir(expert_type_dir) if os.path.isdir(os.path.join(expert_type_dir, d))]
            if not subdirs:
                missing_experts.append(f"{name} (no run folders)")
                continue

            # Sort by string (YYYY-MM-DD...) works well
            latest_run_dir_name = sorted(subdirs)[-1]
            latest_run_path = os.path.join(expert_type_dir, latest_run_dir_name)
            print(f"[INFO] Selected latest run for expert '{name}': {latest_run_dir_name}")

            # Find latest model inside
            files = [f for f in os.listdir(latest_run_path) if f.startswith("model_") and f.endswith(".pt")]
            if not files:
                missing_experts.append(f"{name} (no checkpoints in {latest_run_dir_name})")
                continue

            latest_ckpt = None
            latest_iter = -1
            for fn in files:
                try:
                    num = int(fn.split("model_")[-1].split(".pt")[0])
                except Exception:
                    num = -1
                if num > latest_iter:
                    latest_iter = num
                    latest_ckpt = os.path.join(latest_run_path, fn)
            moe_expert_paths.append(latest_ckpt)

        if missing_experts:
            raise ValueError(f"Missing the following experts for Stage 2 merge: {missing_experts}. Looked in: {expert_pool_dir}/{'{name}'}/*")

        print(f"[INFO] Found pretrained expert checkpoints: {moe_expert_paths}")

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs) if self.privileged_obs_type else obs
        num_privileged_obs = privileged_obs.shape[1]

        # Infer Architecture from first expert
        if moe_expert_paths:
            print(f"[Stage B] Inferring expert architecture from: {moe_expert_paths[0]}")
            inferred_actor_dims, inferred_critic_dims = self._derive_expert_arch(moe_expert_paths[0])
            if inferred_actor_dims:
                print(f"[Stage B] Overriding expert_hidden_actor to {inferred_actor_dims} (matched checkpoint)")
                self.policy_cfg["expert_hidden_actor"] = inferred_actor_dims
            else:
                print(f"[Stage B] Could not infer actor dims from checkpoint. Using config defaults.")

            if inferred_critic_dims:
                print(f"[Stage B] Overriding expert_hidden_critic to {inferred_critic_dims} (matched checkpoint)")
                self.policy_cfg["expert_hidden_critic"] = inferred_critic_dims

        # Build MoE policy afresh, merge experts
        self.policy = ActorCriticMoE(
            num_obs,
            num_privileged_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        # Rebuild algorithm with MoE policy
        self.alg = MOEPPO(self.policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Load weights into MoE
        # Assumes the order in valid_experts matches strict order in ActorCriticMoE experts list.
        # We need access to the policy. In RSL-RL runner: self.alg.actor_critic -> likely 'policy'
        policy = self.alg.policy  # type: ActorCriticMoE

        # Verify it is MoE
        if not hasattr(policy, "actor_moe"):
            raise RuntimeError("Loaded policy is not ActorCriticMoE! Check config.")

        # Load chunks
        for i, ckpt_path in enumerate(moe_expert_paths):
            print(f"[INFO] Loading expert {i} ({expected_experts[i]}) from {ckpt_path}")
            # The loaded dict is a full state_dict of a single-expert policy (likely standard ActorCritic or MoE with 1 expert??)
            # Stage 1 trains standard ActorCritic? Or MoE with 1 expert?
            # User said "expert_straight", "expert_turn".
            # If Stage 1 ran 'OnPolicyMoERunner', it used 'ActorCriticMoE' with num_experts=1?
            # Let's assume the checkpoint contains keys like "actor.0.weight" etc OR "actor_moe.experts.0..."
            # IF Stage 1 used standard ActorCritic, keys are "actor.0.weight".
            # IF Stage 1 used MoE with 1 expert, keys are "actor_moe.experts.0...".

            # We need to map weights to policy.actor_moe.experts[i] and policy.critic_moe.experts[i]

            # Helper to extracting weights for one MLP and putting into another
            # We assume specialized experts are standard MLPs or matching architecture.
            self._load_expert_weights(policy, i, torch.load(ckpt_path, map_location=self.device))

        # Freeze experts to keep specialization
        policy.freeze_experts()
        print("[INFO] Experts loaded and frozen. Training gating network only.")

        # Re-init storage for fine-tuning
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Ensure parameters in-sync for DDP
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Optional: snapshot code state once
        if self.log_dir is not None and not self.disable_logs:
            git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
            if getattr(self, "logger_type", "tensorboard") in ["wandb", "neptune"] and git_file_paths:
                for path in git_file_paths:
                    self.writer.save_file(path)

        # Fine-tuning loop (same structure as OnPolicyRunner.learn)
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs) if self.privileged_obs_type else obs
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        # Buffers
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Act
                    actions = self.alg.act(obs, privileged_obs)
                    # Step
                    obs, rew, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rew, dones = obs.to(self.device), rew.to(self.device), dones.to(self.device)
                    # Normalize obs
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(infos["observations"][self.privileged_obs_type].to(self.device))
                    else:
                        privileged_obs = obs
                    # Process
                    self.alg.process_env_step(rew, dones, infos)

                    # Intrinsic rewards for logging
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Book-keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        if self.alg.rnd:
                            cur_ereward_sum += rew
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rew + intrinsic_rewards
                        else:
                            cur_reward_sum += rew
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # =====================================================================
    # Logging/save/load (ported from OnPolicyRunner)
    # =====================================================================

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Tensorboard/W&B/Neptune logging and pretty terminal print."""
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0].keys():
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # Logging gating stats
            if hasattr(self.alg.policy, "get_gate_info"):
                gate_info = self.alg.policy.get_gate_info()
                total_num_steps = locs["it"] * self.num_steps_per_env * self.env.num_envs
                for k, v in gate_info.items():
                    self.writer.add_scalar(f"MoE/{k}", v, total_num_steps)
                # Print weights occasionally
                if "expert_0_weight" in gate_info:
                    w_str = " | ".join([f"E{i}: {gate_info.get(f'expert_{i}_weight', 0):.2f}" for i in range(len(self.valid_experts))])
                    print(f"[MoE Weights] {w_str}")

            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """Save policy/optimizer and optional RND/normalizers (same structure as OnPolicyRunner)."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(saved_dict, path)

        if getattr(self, "logger_type", "tensorboard") in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        """Load checkpoint (policy/optimizer + RND + normalizers)."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """Return policy callable with (optional) empirical normalization."""
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)

            def policy(x): return self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        """Switch modules to train mode."""
        self.alg.policy.train()
        if self.alg.rnd:
            self.alg.rnd.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """Switch modules to eval mode."""
        self.alg.policy.eval()
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    # =====================================================================
    # Multi-GPU
    # =====================================================================

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size or self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError("Invalid rank/world_size configuration.")

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)
