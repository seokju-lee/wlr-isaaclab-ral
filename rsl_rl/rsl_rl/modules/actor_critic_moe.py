# Copyright (c) 2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
from rsl_rl.networks import Memory


class ActorCriticMoE(ActorCritic):
    """MoE version of ActorCritic that matches ActorCritic's public API.

    API compatibility:
      - act(obs) -> Tensor(actions)          # sample from Normal(mean,std)
      - evaluate(critic_obs) -> Tensor(vals) # critic only
      - act_inference(obs) -> Tensor(mean)   # deterministic mean
      - get_actions_log_prob(actions)        # uses self.distribution set by update_distribution
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # MoE config
        num_experts: int = 3,
        expert_hidden_actor=(256, 256),
        expert_hidden_critic=(256, 256),
        gate_hidden: int = 128,
        activation: str = "elu",
        aux_gate_entropy_coef: float = 0.01,  # CHANGED: Positive to reward entropy (exploration)
        aux_load_balancing_coef: float = 2.0,  # CHANGED: Stronger penalty to break [1,0,0] collapse
        # RR-MoE args
        use_residual: bool = False,  # CHANGED: Disabled by default for MoE isolation test
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        # Transformer Gating args
        use_transformer_gate: bool = False,
        transformer_heads: int = 4,
        transformer_layers: int = 2,  # Compromise: 3 caused OOM, 1 is too simple
        transformer_dropout: float = 0.0,
        # keep ActorCritic kwargs compatible (noise, etc.)
        **kwargs,
    ):
        # Initialize base (builds dummy actor/critic, noise params, etc.)
        # Note: We don't change num_actor_obs here for base init explicitly because base is just a shell.
        # But if we use recurrence, we might want to follow ActorCriticRecurrent pattern?
        # Actually base ActorCritic doesn't do much with obs dim except building MLPs which we replace/ignore.
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            **kwargs,
        )

        if expert_hidden_actor == (256, 256) and "actor_hidden_dims" in kwargs:
            expert_hidden_actor = kwargs["actor_hidden_dims"]
        if expert_hidden_critic == (256, 256) and "critic_hidden_dims" in kwargs:
            expert_hidden_critic = kwargs["critic_hidden_dims"]

        self.is_recurrent = True  # RR-MoE is recurrent by default (for gating)
        self.use_residual = use_residual

        # Replace base actor/critic MLPs with MoE blocks, but keep the same attributes.
        act_mod = resolve_nn_activation(activation) if isinstance(activation, str) else activation
        self._act = act_mod if isinstance(act_mod, nn.Module) else act_mod()

        # --- Recurrent Memory (Gating + Residual) ---
        # Actor memory: processes obs -> feeds Gating Network AND Residual Network
        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        # Critic memory: processes critic_obs -> feeds Critic MoE
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        print(f"[RR-MoE] Initialized. Residual={use_residual}, RNN={rnn_type}, Hidden={rnn_hidden_dim}")

        # --- Expert Modules ---
        # Experts take RAW observation? Or Memory output?
        # User goal: "Freeze Experts". Pretrained experts expect RAW observation (MLP).
        # So Experts must take original 'obs', NOT rnn features.
        # BUT Gating Network needs Memory features to be "Recurrent Gating".
        # AND Residual Network usually takes Memory features (to be smart).

        self.actor_moe = _MoEActor(num_actor_obs, num_actions, num_experts, expert_hidden_actor, self._act)
        # 2. Critic Architecture -> GLOBAL MLP (Standard)
        # Replaces MoE Critic to avoid bootstrapping crashes and gating mismatch.
        # Simple MLP that learns V(s) for the mixed policy.
        critic_layers = []
        critic_layers.append(nn.Linear(rnn_hidden_dim, expert_hidden_critic[0]))
        critic_layers.append(self._new_act())
        for i in range(len(expert_hidden_critic) - 1):
            critic_layers.append(nn.Linear(expert_hidden_critic[i], expert_hidden_critic[i + 1]))
            critic_layers.append(self._new_act())
        critic_layers.append(nn.Linear(expert_hidden_critic[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Removed MoE Critic
        # self.critic_moe = _MoECritic(num_critic_obs, num_experts, expert_hidden_critic, self._act)

        # --- Gating Networks (Input: RNN features) ---
        # "Transformer-based Time-Aware Gating (T-Tag)"
        # We replace the simple MLP gate with a Transformer that attends State <-> Experts
        self.use_transformer_gate = use_transformer_gate

        if self.use_transformer_gate:
            # 1. Expert Embeddings (Key/Value for attention)
            # Learnable representation of each expert's capability
            self.expert_embeddings = nn.Parameter(torch.randn(num_experts, rnn_hidden_dim))

            # 2. Transformer Encoder (Self-Attention)
            # Input sequence: [StateToken, ExpertToken_1, ..., ExpertToken_N]
            # d_model = rnn_hidden_dim
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=rnn_hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=rnn_hidden_dim * 2,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True
            )
            self.gate_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False)

            # 3. Projection to Scalar Score
            # Maps processed Expert Token -> Gating Logit
            self.gate_out = nn.Linear(rnn_hidden_dim, 1)

            print(f"[Transformer-MoE] Gating via Transformer ({transformer_layers} layers, {transformer_heads} heads).")
        else:
            # Fallback to MLP
            self.gate_actor = nn.Sequential(
                nn.Linear(rnn_hidden_dim, gate_hidden),
                self._new_act(),
                nn.Linear(gate_hidden, num_experts),
            )

        # Value Gating (Critic): NOT NEEDED for Global MLP Critic.
        # self.gate_value = ... (Removed)
        self.gate_value = None

        # --- Residual Actor (Input: RNN features) ---
        if self.use_residual:
            # Simple MLP correction
            self.residual_actor = nn.Sequential(
                nn.Linear(rnn_hidden_dim, 128),
                self._new_act(),
                nn.Linear(128, 128),
                self._new_act(),
                nn.Linear(128, num_actions)
            )
            # Initialize residual to near-zero to start with expert behavior
            # Last layer weights small
            with torch.no_grad():
                self.residual_actor[-1].weight.mul_(0.01)
                self.residual_actor[-1].bias.zero_()
        else:
            self.residual_actor = None

        # Log-std for actor distribution (reuse std/log_std from base if provided)
        # If base created self.std or self.log_std, keep using it. Otherwise keep a log_std here.
        if not hasattr(self, "std") and not hasattr(self, "log_std"):
            self.log_std_moe = nn.Parameter(torch.zeros(num_actions))
        else:
            self.log_std_moe = None

        # Expert STDs (for Std Mixing)
        self.expert_log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(num_actions)) for _ in range(num_experts)
        ])

        # Learnable global correction for expert STDs.
        # Experts might be too noisy (e.g. 1.87), so we allow the MoE to learn to reduce it.
        # final_std = mixed_expert_std * exp(correction)
        # INIT: -1.0 to start safe (~0.69 effective std) preventing -300 reward crash.
        self.log_std_correction = nn.Parameter(torch.ones(num_actions) * -1.0)

        # Aux regularization
        self.gate_entropy_coef = aux_gate_entropy_coef
        self.load_balancing_coef = aux_load_balancing_coef
        self._last_gate_entropy = torch.tensor(0.0)
        self._last_load_balance_loss = torch.tensor(0.0)
        self._last_gate_weights = None
        self._gate_weights_for_loss = None

        # Keep ActorCritic.actor/critic pointing somewhere valid for tooling/debug APIs.
        self.actor = nn.Identity()
        # self.critic = nn.Identity() # This is now the global MLP critic

        # Disable arg checks for speed (as in base)
        Normal.set_default_validate_args(False)

    # -------- API helpers to mirror ActorCritic --------
    def _new_act(self):
        return self._act.__class__() if isinstance(self._act, nn.Module) else self._act()

    def _actor_mean(self, obs: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        is_2d_input = obs.ndim == 2

        # 1. Update/Use Memory (Recurrent)
        # memory_a returns features [Batch, rnn_hidden_dim]
        # It handles internal state update if masks is None.
        # features: [Time, Batch, Hidden] or [Batch, Hidden]

        # Handle 3D input without masks (e.g. from act_inference on batch)
        if obs.ndim == 3 and masks is None:
            # Assume [Time, Batch, Dim] - Create dummy masks/hidden states
            T, B, _ = obs.shape
            device = obs.device
            masks = torch.ones(T, B, device=device, dtype=torch.bool)  # All valid
            # Initialize zero hidden states [Layers, B, Hidden]
            # Need to know num_layers/hidden_dim from memory module
            h_0 = torch.zeros(self.memory_a.rnn.num_layers, B, self.memory_a.rnn.hidden_size, device=device)
            if isinstance(self.memory_a.rnn, nn.LSTM):
                c_0 = torch.zeros_like(h_0)
                hidden_states = (h_0, c_0)
            else:
                hidden_states = h_0
        features = self.memory_a(obs, masks, hidden_states)

        if masks is not None:
            obs = unpad_trajectories(obs, masks)

        # Flatten Time & Batch dimensions for Gating/Experts
        features_original_shape = None
        if features.ndim == 3:
            T, B, D = features.shape
            features_original_shape = (T, B)
            features = features.reshape(T * B, D)
            # Must also flatten obs because Experts use it
            obs = obs.reshape(T * B, -1)

        # 2. Gating (Cognition)
        if self.use_transformer_gate:
            # Transformer Gating Logic
            # Construct Sequence: [Batch, 1+N, Dim]  (State + Experts)
            batch_size = features.shape[0]

            # (a) State Token: [Batch, 1, Dim]
            state_token = features.unsqueeze(1)

            # (b) Expert Tokens: [Batch, N, Dim]
            # Expand learnable embeddings to batch size
            expert_tokens = self.expert_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

            # (c) Concat: [Batch, 1+N, Dim]
            transformer_input = torch.cat([state_token, expert_tokens], dim=1)

            # (d) Pass through Transformer
            # batch_first=True, so [Batch, Seq, Dim]
            # transformer_input = transformer_input.contiguous() # Already contiguous from concat generally

            # Force Math SDPA to avoid CUDA config errors with Flash/MemEfficient on some setups
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                out_seq = self.gate_transformer(transformer_input)

            # (e) Extract Processed Expert Tokens (Indices 1 to End)
            # The State Token (Index 0) has now attended to experts and gathered info,
            # and Experts have attended to State.
            processed_experts = out_seq[:, 1:, :]  # [Batch, N, Dim]

            # (f) Project to Logits: [Batch, N, 1] -> [Batch, N]
            logits = self.gate_out(processed_experts).squeeze(-1)

        else:
            # MLP Gating
            logits = self.gate_actor(features)

        weights = F.softmax(logits, dim=-1)

        # Store for logging
        with torch.no_grad():
            ent = (-weights * (weights.clamp_min(1e-8).log())).sum(dim=-1).mean()
            self._last_gate_entropy = ent
            # Store full tensor for histogram logging (detached)
            self._last_gate_weights = weights.detach()

        # Store ATTACHED tensor for loss calculation
        self._gate_weights_for_loss = weights

        # DEBUG: Print Gating Weights occasionally
        if torch.rand(1) < 0.001:  # approx every 1000 steps per env
            print(f"[DEBUG] Gating Weights Mean: {weights.mean(dim=0).detach().cpu().numpy()}")

        # 3. Expert Execution (Skill) - Experts use RAW 'obs' (Pretrained on raw obs)
        expert_action = self.actor_moe(obs, weights)

        # 4. Residual Correction (Adaptation) based on History
        if self.use_residual:
            residual = self.residual_actor(features)
            mean = expert_action + residual
        else:
            mean = expert_action

        # Should not really happen if Memory output 3D, but safety
        if is_2d_input and mean.ndim == 3 and mean.shape[0] == 1:
            mean = mean.squeeze(0)

        return mean

    def _critic_value(self, critic_obs: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        is_2d_input = critic_obs.ndim == 2
        # Critic also uses memory for consistency (and usually critic needs memory more than actor)
        # Critic also uses memory for consistency (and usually critic needs memory more than actor)
        features = self.memory_c(critic_obs, masks, hidden_states)

        if masks is not None:
            critic_obs = unpad_trajectories(critic_obs, masks)

        orig_shape = None
        if features.ndim == 3:
            T, B, D = features.shape
            orig_shape = (T, B)
            features = features.reshape(T * B, D)
            critic_obs = critic_obs.reshape(T * B, -1)

        # Global MLP Critic Forward
        # Simple V(s) prediction from features
        value = self.critic(features)

        # Safety check for 2D input -> 3D output
        if is_2d_input and value.ndim == 3 and value.shape[0] == 1:
            value = value.squeeze(0)

        return value

    # -------- Recurrent Interface --------
    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    # -------- ActorCritic-compatible interface --------
    def update_distribution(self, observations: torch.Tensor, masks=None, hidden_states=None):
        # Handle recurrence args
        mean = self._actor_mean(observations, masks, hidden_states)
        # ... logic for std ...
        # Std Mixing Strategy (Priority High):
        # Always mix expert STDs if available, regardless of base noise config.
        # This fixes the issue where default "scalar" type blocked mixing.
        if hasattr(self, "expert_log_stds") and self._gate_weights_for_loss is not None:
            # [NumExperts, NumActions]
            all_expert_stds = torch.stack([torch.exp(ls) for ls in self.expert_log_stds])
            # [Batch, NumExperts] @ [NumExperts, NumActions] -> [Batch, NumActions]
            # We use accurate weights from the current forward pass
            # [Batch, NumExperts] @ [NumExperts, NumActions] -> [Batch, NumActions]
            # We use accurate weights from the current forward pass
            mixed_std = self._gate_weights_for_loss @ all_expert_stds

            # Apply learnable correction: allow model to dampen (or boost) the expert noise
            std = mixed_std * torch.exp(self.log_std_correction)
        elif hasattr(self, "noise_std_type") and self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif hasattr(self, "noise_std_type") and self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            # fallback to MoE-local log_std if base noise params are not present
            std = torch.exp(self.log_std_moe).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, masks=None, hidden_states=None, **kwargs) -> torch.Tensor:
        self.update_distribution(observations, masks, hidden_states)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        # act_inference usually doesn't pass masks/hidden_states in arguments in some runners?
        # But base ActorCritic.act_inference only takes obs.
        # ActorCriticRecurrent.act_inference only takes obs (uses internal state).
        return self._actor_mean(observations)

    def evaluate(self, critic_observations: torch.Tensor, masks=None, hidden_states=None, **kwargs) -> torch.Tensor:
        # value-only path
        return self._critic_value(critic_observations, masks, hidden_states)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # identical to base class (distribution set in update_distribution)
        return super().get_actions_log_prob(actions)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Check for shape mismatch: Distribution (T, B, A) vs Actions (N, A)
        # Verify 3D distribution vs 2D actions
        if self.distribution.loc.ndim == 3 and actions.ndim == 2:
            # Flatten distribution to match actions
            flat_loc = self.distribution.loc.reshape(-1, self.distribution.loc.shape[-1])
            flat_scale = self.distribution.scale.reshape(-1, self.distribution.scale.shape[-1])
            flat_dist = Normal(flat_loc, flat_scale)
            return flat_dist.log_prob(actions).sum(dim=-1)

        return super().get_actions_log_prob(actions)

    def regularization_loss(self):
        # 1. Entropy Loss: Encourage high gate entropy PER SAMPLE (uncertainty)
        # Negative sign: larger entropy -> smaller loss
        entropy_loss = -self.gate_entropy_coef * self._last_gate_entropy

        # 2. Load Balancing Loss: Encourage UNIFORM usage across the batch
        # We want mean_weights to be close to 1/num_experts
        if self._gate_weights_for_loss is not None:
            # [NumExperts]
            # Use the attached tensor so gradients flow back to the gate
            mean_usage = self._gate_weights_for_loss.mean(dim=0)

            # Robustly get num_experts (avoid shape[0] crash if mean_usage is scalar)
            num_experts = len(self.actor_moe.experts)
            target_usage = torch.ones_like(mean_usage) / num_experts

            # MSE between actual usage and uniform usage
            lb_loss = (mean_usage - target_usage).pow(2).mean()
            self._last_load_balance_loss = lb_loss
            # Note: We return self._last_load_balance_loss which is now attached (if calculation was attached)
            # Actually self._last_load_balance_loss = lb_loss assigns the variable.
            # But the return statement creates a new graph node.
        else:
            lb_loss = torch.tensor(0.0, device=self.device if hasattr(self, "device") else "cpu")

        return entropy_loss + self.load_balancing_coef * lb_loss

    def freeze_experts(self):
        """Freeze the weights of the expert networks (actor and critic)."""
        print("[ActorCriticMoE] Freezing expert weights.")
        for param in self.actor_moe.parameters():
            param.requires_grad = False
            param.requires_grad = False
        # Critic is now Global MLP and TRAINABLE. Do not freeze.
        # for param in self.critic_moe.parameters():
        #     param.requires_grad = False
        # Also freeze expert STDs
        for param in self.expert_log_stds.parameters():
            param.requires_grad = False

    def get_gate_info(self):
        """Return dict of gating stats (mean entropy, mean weights per expert)."""
        info = {
            "gate_entropy": self._last_gate_entropy.item(),
            "load_balance_loss": self._last_load_balance_loss.item(),
        }
        if self._last_gate_weights is not None:
            # 1. Scalar means for easy tracking
            # _last_gate_weights is [Batch, NumExperts]
            mean_weights = self._last_gate_weights.mean(dim=0).reshape(-1)
            ws = mean_weights.cpu().numpy().tolist()
            for i, w in enumerate(ws):
                info[f"expert_{i}_weight"] = w

            # 2. Full tensor for histogram logging
            info["gate_weights"] = self._last_gate_weights

        return info


class _MoEActor(nn.Module):
    """Actor-side MoE that produces action means."""

    def __init__(self, obs_dim, act_dim, num_experts, hidden_dims, act_mod):
        super().__init__()
        self._act_mod = act_mod

        def _new_act():
            return self._act_mod.__class__() if isinstance(self._act_mod, nn.Module) else self._act_mod()

        self.experts = nn.ModuleList()
        for _ in range(int(num_experts)):
            layers = []
            in_dim = obs_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_dim, h), _new_act()]
                in_dim = h
            layers += [nn.Linear(in_dim, act_dim)]
            self.experts.append(nn.Sequential(*layers))

    def forward(self, obs: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        # expert outputs: [E][B, A] -> stack -> [B, A, E]
        outs = torch.stack([expert(obs) for expert in self.experts], dim=-1)
        # weighted sum over experts -> [B, A]
        return torch.einsum("bae,be->ba", outs, gate_weights)


class _MoECritic(nn.Module):
    """Critic-side MoE that produces scalar values."""

    def __init__(self, obs_dim, num_experts, hidden_dims, act_mod):
        super().__init__()
        self._act_mod = act_mod

        def _new_act():
            return self._act_mod.__class__() if isinstance(self._act_mod, nn.Module) else self._act_mod()

        self.experts = nn.ModuleList()
        for _ in range(int(num_experts)):
            layers = []
            in_dim = obs_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_dim, h), _new_act()]
                in_dim = h
            layers += [nn.Linear(in_dim, 1)]
            self.experts.append(nn.Sequential(*layers))

    def forward(self, obs: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        # values per expert: [E][B,1] -> [B,1,E]
        vals = torch.stack([e(obs) for e in self.experts], dim=-1)
        # weighted sum over experts: [B,1,E] * [B,E] -> [B,1,E], then sum on experts -> [B,1]
        weighted = (vals * gate_weights.unsqueeze(1)).sum(dim=-1)
        return weighted  # [B,1]
