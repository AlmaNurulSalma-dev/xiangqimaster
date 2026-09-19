# 05 — Neural Network Specification

> **Read this when:** implementing or modifying the Policy-Value network — its layers, tensor shapes, the two output heads, and the training targets. This network is the shared brain used by the PPO, IL+RL, and MCTS agents.

---

## 1. Purpose

The neural network is a **Policy-Value network** — a single network with two outputs:
1. **Policy head** — predicts a probability distribution over all possible moves ("which move should I play?")
2. **Value head** — predicts the expected game outcome from the current position ("am I winning?")

This dual-head design is the core of the AlphaZero paradigm. The same architecture is reused across:
- Imitation Learning (Agent 2 Phase 1) — trained to predict human moves + outcomes
- PPO (Agent 1, Agent 2 Phase 2) — the policy is the actor, the value is the critic
- MCTS (Agent 3) — the policy guides tree search priors, the value replaces random rollouts

---

## 2. Input

- **Shape:** 14 × 10 × 9 (channels × height × width)
- **Content:** the player-relative board state tensor from 03-ENVIRONMENT.md
- **Values:** binary (0.0 or 1.0)
- **Batching:** input to the network is (batch_size, 14, 10, 9)

---

## 3. Overall Architecture

The network follows the AlphaZero-style structure: a shared convolutional "body" (residual tower) that splits into two heads.

```
Input (14 × 10 × 9)
      │
      ▼
┌─────────────────────┐
│  Input Convolution  │  Conv2D(14 → C) + BatchNorm + ReLU
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Residual Tower     │  N residual blocks (default N = 10)
│  (the "body")       │  each preserves shape (C × 10 × 9)
└─────────────────────┘
      │
      ├──────────────────────┐
      ▼                      ▼
┌──────────────┐      ┌──────────────┐
│  Policy Head │      │  Value Head  │
└──────────────┘      └──────────────┘
      │                      │
      ▼                      ▼
policy logits          value scalar
(2086 values)          (1 value in [-1,1])
```

---

## 4. Component Specifications

### 4.1 Input Convolution Block
- Conv2D: 14 input channels → C output channels, kernel 3×3, padding 1 (preserves 10×9 spatial size)
- BatchNorm2D
- ReLU activation
- **C (channel width):** default 64. Can increase to 128 or 256 for a larger, stronger network (at higher compute cost). Make it a config value.

### 4.2 Residual Block
Each residual block preserves the tensor shape (C × 10 × 9) and consists of:
- Conv2D: C → C, kernel 3×3, padding 1
- BatchNorm2D
- ReLU
- Conv2D: C → C, kernel 3×3, padding 1
- BatchNorm2D
- **Skip connection:** add the block's input to its output (before the final ReLU)
- ReLU

**Number of blocks (N):** default 10. AlphaZero used 19–40. For a thesis with limited compute, 10 is a reasonable balance. Config value — can be reduced to 6 for faster iteration or increased for final runs.

### 4.3 Policy Head
Converts the body output into move probabilities:
- Conv2D: C → 2 channels, kernel 1×1
- BatchNorm2D
- ReLU
- Flatten (2 × 10 × 9 = 180 features)
- Linear: 180 → 2086
- Output: **raw logits** over the 2,086 possible moves (softmax applied later, after masking)

**Important:** the policy head outputs raw logits, NOT probabilities. Masking of illegal moves and softmax happen outside the network (in the agent), so illegal moves can be set to −infinity before softmax.

### 4.4 Value Head
Converts the body output into a single scalar:
- Conv2D: C → 1 channel, kernel 1×1
- BatchNorm2D
- ReLU
- Flatten (1 × 10 × 9 = 90 features)
- Linear: 90 → 256
- ReLU
- Linear: 256 → 1
- Tanh activation → output in [−1, 1]

**Interpretation:** −1 = certain loss, 0 = even, +1 = certain win, from the current player's perspective.

---

## 5. Output Summary

| Output | Shape | Activation | Meaning |
|---|---|---|---|
| Policy logits | (batch, 2086) | none (raw logits) | Move preferences (masked + softmaxed externally) |
| Value | (batch, 1) | tanh | Expected outcome in [−1, 1] |

---

## 6. Training Targets & Loss Functions

The network is trained differently depending on the agent, but the loss structure is similar.

### 6.1 Imitation Learning (Agent 2 Phase 1)
- **Policy target:** the action index of the human's move → cross-entropy loss between predicted policy and the one-hot human move
- **Value target:** the game's final outcome from this position's perspective → MSE loss between predicted value and actual outcome
- **Total loss:** `policy_loss + value_loss` (optionally weighted)

### 6.2 MCTS Training (Agent 3)
- **Policy target:** the MCTS visit-count distribution (a soft target, not one-hot) → cross-entropy
- **Value target:** the self-play game outcome → MSE
- **Total loss:** `policy_loss + value_loss + L2 regularization`

### 6.3 PPO (Agent 1, Agent 2 Phase 2)
- The network is used as actor (policy) and critic (value)
- Loss is the PPO clipped surrogate objective + value loss + entropy bonus (see 06-AGENTS.md and 07-TRAINING.md for the full PPO loss)

### 6.4 Regularization
- **L2 weight decay:** default 1e-4, applied to all weights (standard in AlphaZero)
- **BatchNorm:** provides implicit regularization
- Dropout is generally NOT used in AlphaZero-style networks (BatchNorm suffices), but MC Dropout could be added to the value head if uncertainty estimation is desired (optional, advanced)

---

## 7. Design Rationale (For Thesis Defense)

Be ready to explain these choices:

- **Why a shared body with two heads?** Efficiency and regularization — the policy and value tasks share useful features (understanding the board position helps both). This is the proven AlphaZero design.
- **Why residual blocks?** They allow training deep networks without vanishing gradients; skip connections preserve information and stabilize training.
- **Why 1×1 convolutions in the heads?** They reduce channel dimensionality cheaply before the fully-connected layers, keeping parameter count manageable.
- **Why tanh on the value?** Because game outcomes are naturally bounded in [−1, +1] (loss to win).
- **Why raw logits from the policy?** So illegal-move masking can be applied before the softmax.

---

## 8. Parameter Count & Compute Notes

- With C=64 and N=10 blocks, the network is relatively small (a few million parameters) — trainable on a single Colab T4 GPU.
- Increasing C or N increases strength but also training time and inference latency (critical for MCTS, which calls the network thousands of times per move).
- **For MCTS especially, keep the network small enough that inference is fast** — a huge network makes MCTS impractically slow. Balance strength vs. speed.

---

## 9. Configurable Hyperparameters (in config)

| Parameter | Default | Notes |
|---|---|---|
| Channel width (C) | 64 | 64/128/256 |
| Number of residual blocks (N) | 10 | 6 (fast) to 19 (strong) |
| Policy head intermediate channels | 2 | rarely changed |
| Value head hidden size | 256 | 128–512 |
| L2 weight decay | 1e-4 | standard |

---

## 10. Testing the Network

- [ ] Input shape (batch, 14, 10, 9) produces policy shape (batch, 2086) and value shape (batch, 1)
- [ ] Value output is always within [−1, 1]
- [ ] Forward pass works on both CPU and GPU
- [ ] Network is deterministic given fixed weights and input (in eval mode)
- [ ] Gradient flows to all layers (no dead parameters)
- [ ] A single training step reduces loss on a tiny fixed batch (sanity check — the network can memorize)

---

*Cross-references: 03-ENVIRONMENT.md (input tensor), 06-AGENTS.md (how each agent uses the network), 07-TRAINING.md (training loops and losses).*
