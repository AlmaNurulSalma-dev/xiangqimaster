# 12 — Glossary

> **Read this when:** any term is unclear. This is the shared vocabulary for the whole project — reinforcement learning, Xiangqi, and technical terms. Terms are grouped by category.

---

## Reinforcement Learning Terms

**Reinforcement Learning (RL)** — A machine learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards, rather than learning from labeled examples. The goal is to maximize cumulative reward.

**Agent** — The decision-maker being trained. In this project, each of the four approaches (PPO, IL+RL, MCTS+NN, Minimax) is an agent.

**Environment** — The world the agent acts in. Here, the Xiangqi game simulator: it holds the board state, accepts moves, and returns rewards.

**State** — A complete description of the current situation. Here, the board position encoded as a 14×10×9 tensor.

**Action** — A choice the agent makes. Here, a move, represented as an integer index (0–2085).

**Reward** — The numerical feedback signal. Here, +1 for winning, −1 for losing, +0.1 for a draw, 0 otherwise.

**Policy** — The agent's strategy: a mapping from states to a probability distribution over actions ("given this board, how likely am I to play each move?").

**Value (function)** — An estimate of how good a state is: the expected future reward from that state. Here, the value head predicts the expected game outcome in [−1, 1].

**Episode** — One complete game from start to terminal state.

**Self-Play** — Training by having the agent play against itself (or copies of itself), generating its own training data without human games.

**Sparse Reward** — A reward given only rarely (here, only at game end). Harder to learn from than dense reward but cleaner.

**Dense Reward / Reward Shaping** — Adding intermediate rewards (here, optional material advantage per move) to give the agent more frequent feedback.

**Exploration vs Exploitation** — The trade-off between trying new actions (exploration) to discover better strategies, and using known-good actions (exploitation). Balanced via entropy bonuses, temperature, and PUCT.

**PPO (Proximal Policy Optimization)** — A popular, stable policy-gradient RL algorithm. It improves the policy while "clipping" updates to prevent destabilizing large changes. Used by Agent 1 and Agent 2.

**MaskablePPO** — A PPO variant (from sb3-contrib) that supports action masking, so the agent never considers illegal moves. Essential for Xiangqi's large action space.

**Policy Gradient** — A family of RL methods that directly adjust the policy in the direction that increases expected reward.

**Advantage** — How much better an action was than the value function expected. Used to focus learning on surprisingly good or bad actions.

**GAE (Generalized Advantage Estimation)** — A technique to estimate advantages that balances bias and variance, controlled by the λ parameter.

**Actor-Critic** — An architecture where an "actor" (policy) chooses actions and a "critic" (value function) evaluates them. The Policy-Value network is an actor-critic in one network.

**Discount Factor (γ)** — How much future rewards are valued relative to immediate ones. Near 1.0 means the agent cares about long-term outcomes (important in a game decided at the end).

**Entropy (bonus)** — A measure of how uncertain/spread-out the policy is. Adding an entropy bonus encourages exploration and prevents premature convergence.

**Curriculum Learning** — Training against progressively harder tasks/opponents rather than the full difficulty at once. Tested in Experiment 4.

**Imitation Learning (IL) / Behavior Cloning** — Learning to copy expert behavior via supervised learning (here, predicting professional players' moves). Used in Agent 2 Phase 1.

**Replay Buffer** — Storage of past experiences (states, actions, outcomes) that training samples from.

---

## MCTS & AlphaZero Terms

**MCTS (Monte Carlo Tree Search)** — A search algorithm that builds a tree of possible future positions by running many simulations, used to choose strong moves. Core of Agent 3.

**Simulation (in MCTS)** — One pass through the four MCTS phases (selection, expansion, evaluation, backpropagation). More simulations per move = stronger but slower.

**Selection (MCTS phase)** — Walking down the tree from the root, choosing children by the PUCT formula, until reaching an unexpanded node.

**Expansion (MCTS phase)** — Adding a new node's children to the tree, using the network's prior probabilities.

**Evaluation (MCTS phase)** — Estimating a position's value. In AlphaZero, the value head provides this directly (no random rollout).

**Backpropagation (MCTS phase)** — Propagating the evaluated value back up the tree, updating visit counts and average values.

**PUCT** — The formula MCTS uses to select children, balancing the average value (Q) with an exploration term based on the network's prior (P) and visit counts. `c_puct` controls the balance.

**Visit Count** — How many simulations passed through a given move. The final move is chosen by highest visit count (a robust signal).

**Prior Probability (P)** — The network policy's initial estimate of how good each move is, before search refines it.

**Dirichlet Noise** — Random noise added to the root priors during training self-play to force exploration of varied openings.

**AlphaZero** — DeepMind's algorithm combining MCTS with a policy-value network, trained purely by self-play. Mastered Chess, Go, Shogi. The template for Agent 3 (and never applied to Xiangqi — the research gap).

**Temperature** — A parameter controlling how greedily moves are chosen from the visit-count distribution. High temperature = more random (exploration); near zero = pick the best (exploitation).

---

## Neural Network Terms

**Policy-Value Network** — A single neural network with two outputs: a policy (move probabilities) and a value (position evaluation). The shared brain of the neural agents.

**Policy Head** — The network branch outputting move preferences (2086 logits).

**Value Head** — The network branch outputting a single scalar in [−1, 1] estimating the outcome.

**Residual Block** — A network building block with a "skip connection" that adds the input to the output, enabling deep networks to train stably.

**Skip Connection** — A shortcut that adds a layer's input to its output, preventing vanishing gradients in deep networks.

**Convolution (Conv2D)** — A neural network operation that detects spatial patterns — ideal for board/grid data.

**Batch Normalization (BatchNorm)** — A technique that stabilizes and speeds up training by normalizing layer activations.

**Backbone / Body** — The shared trunk of the network (here, the residual tower) that both heads branch from.

**Logits** — Raw, unnormalized network outputs before applying softmax. The policy head outputs logits so illegal moves can be masked before softmax.

**Softmax** — A function converting logits into a probability distribution (values summing to 1).

**Cross-Entropy Loss** — A loss function for classification/policy targets, measuring the difference between predicted and target distributions.

**MSE (Mean Squared Error)** — A loss function for regression/value targets, measuring squared difference between prediction and target.

**Weight Decay (L2 Regularization)** — A penalty on large weights that reduces overfitting.

**Epoch** — One full pass through the training dataset.

**Batch** — A group of examples processed together before one weight update.

**Learning Rate** — How big a step the optimizer takes when updating weights.

**Transfer Learning** — Using a model trained on one task as a starting point for another. (Here, IL pre-training initializes RL.)

---

## Xiangqi Terms

**Xiangqi (象棋)** — Chinese Chess. A two-player perfect-information strategy game on a 9×10 board.

**General / King (將/帥)** — The piece that must be protected; checkmating it wins. Confined to the palace.

**Advisor / Guard (士/仕)** — Moves one point diagonally, confined to the palace.

**Elephant / Minister (象/相)** — Moves two points diagonally, cannot cross the river, blocked at the "eye".

**Horse (馬)** — Moves in an L-shape but can be blocked at the "leg" (unlike the Chess knight).

**Chariot / Rook (車)** — Moves any distance orthogonally; the strongest piece.

**Cannon (砲/炮)** — Moves like a chariot but must jump exactly one piece to capture. Xiangqi's most distinctive piece.

**Soldier / Pawn (卒/兵)** — Moves forward; gains sideways movement after crossing the river; never moves backward.

**River (河界)** — The horizontal divide across the board's middle; affects Elephant and Soldier movement.

**Palace (九宫)** — The 3×3 zone confining the General and Advisors.

**Flying General (飞将)** — The rule forbidding the two Generals from directly facing each other on an open file.

**Check (将军)** — A threat to capture the General on the next move; must be answered.

**Checkmate (将死)** — In check with no legal escape; a loss.

**Stalemate (困毙)** — No legal moves (but not in check); in Xiangqi this is a LOSS (unlike Chess).

**Ply** — One half-move (one player's single move). A "move" in common speech is often two plies.

**WXF Notation** — The World Xiangqi Federation move notation used in the dataset.

**Opening Theory (开局)** — Established sequences of strong opening moves. Examples: 当头炮 (Central Cannon), 顺炮 (Same-Direction Cannons), 飞相局 (Flying Elephant).

**ElephantEye (象眼)** — A well-known open-source Xiangqi engine (`xqbase/eleeye`, LGPL-2.1, C++, compile-from-source), used as a benchmark opponent. **Pikafish** (Stockfish-based) and **Fairy-Stockfish** are stronger, actively-maintained alternatives that speak the same UCCI protocol.

---

## Evaluation & Statistics Terms

**Elo Rating** — A number representing playing strength; rating differences predict expected match outcomes. The primary metric.

**Win Rate** — Fraction of games won against a given opponent.

**Confidence Interval (95%)** — A range that likely contains the true value; reported with Elo to convey uncertainty.

**Round-Robin** — A tournament where every participant plays every other participant.

**KL Divergence** — A measure of how different one probability distribution is from another; used in Experiment 3 to compare opening distributions.

**Structural Hamming Distance** — (If used) a measure of difference between two graphs; more relevant to causal work — not central here.

**Ablation Study** — Removing or varying one component to measure its contribution (e.g., Experiment 5 varies MCTS simulation count).

**Baseline** — A simple reference method (here, Minimax) that the more advanced methods are compared against.

---

## Software & Tooling Terms

**Gymnasium (Gym)** — The standard Python interface for RL environments (reset/step API). The environment conforms to it.

**stable-baselines3 / sb3-contrib** — RL libraries providing PPO / MaskablePPO implementations.

**PyTorch** — The deep learning framework used for all networks and training.

**Streamlit** — The Python framework for building the interactive dashboard.

**Weights & Biases (wandb)** — An experiment tracking tool for logging training metrics.

**UCCI (Universal Chinese Chess Interface)** — The text protocol for communicating with Xiangqi engines (used to talk to ElephantEye, Pikafish, Fairy-Stockfish). It is the Xiangqi counterpart of **UCI** (Universal Chess Interface), which is used for international chess — the two are related but distinct; Xiangqi uses UCCI.

**Checkpoint** — A saved snapshot of model weights (and training state) allowing training to resume.

**Colab (Google Colab)** — A cloud notebook service providing free/cheap GPU access, used for training.

**Action Masking** — Setting illegal actions' probabilities to zero (logits to −∞) so the agent only chooses legal moves.

---

*This glossary is the shared vocabulary. When a doc uses a term defined here, it means exactly this. If a needed term is missing, add it here rather than defining it inconsistently elsewhere.*
