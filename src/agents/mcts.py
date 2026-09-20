"""Monte Carlo Tree Search core for the AlphaZero-style agent (docs/06 5).

Each move decision runs N simulations. A simulation has four phases:

1. **Selection** — from the root, descend by the PUCT score
   ``-Q(child) + c_puct * P(child) * sqrt(ΣN) / (1 + N(child))``
   until reaching an unexpanded leaf. (``-Q`` because a child's stored value is
   from the child-to-move's perspective, i.e. the opponent of the parent.)
2. **Expansion** — evaluate the leaf with the network to get masked priors for
   the legal moves, and create the children.
3. **Evaluation** — use the network's value head for the leaf (no rollout).
   A leaf with no legal moves is terminal: the side to move has lost → value -1.
4. **Backpropagation** — walk back up, adding the value and negating it at each
   step (negamax), incrementing visit counts.

The move is then chosen by visit count (more robust than raw value).

This is a clear, single-threaded reference implementation. Batching leaf
evaluations (docs/07 section 4.4) is a later optimization.
"""

from __future__ import annotations

import numpy as np
import torch

from src.environment import action_space, encoder
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.models.network import PolicyValueNetwork
from src.utils.config import (
    MCTS_C_PUCT,
    MCTS_DIRICHLET_ALPHA,
    MCTS_DIRICHLET_EPSILON,
    MCTS_SIMULATIONS,
)


class Node:
    """One node (a board position) in the search tree."""

    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        """Mean value from this node's side-to-move perspective (0 if unvisited)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@torch.no_grad()
def _evaluate(
    network: PolicyValueNetwork,
    board: Board,
    legal_moves: list,
) -> tuple[dict[int, float], float]:
    """Return (priors over legal action indices, value) from the network."""
    observation = encoder.encode(board)
    tensor = torch.from_numpy(observation).unsqueeze(0)
    logits, value = network(tensor)
    logits = logits.squeeze(0).cpu().numpy()

    indices = [action_space.move_to_index(m) for m in legal_moves]
    legal_logits = logits[indices]
    legal_logits = legal_logits - legal_logits.max()  # numerical stability
    weights = np.exp(legal_logits)
    probs = weights / weights.sum()
    priors = {idx: float(p) for idx, p in zip(indices, probs)}
    return priors, float(value.item())


def _select_child(node: Node, c_puct: float) -> tuple[int, "Node"]:
    """Pick the child maximizing the PUCT score."""
    sqrt_total = np.sqrt(node.visit_count)
    best_score = -np.inf
    best_action = -1
    best_child: Node | None = None
    for action, child in node.children.items():
        q = -child.value()  # child value is from the opponent's perspective
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_child is not None
    return best_action, best_child


def _backpropagate(path: list[Node], value: float) -> None:
    """Update visit counts and values back up the path (negamax)."""
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _add_dirichlet_noise(
    priors: dict[int, float],
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Mix Dirichlet noise into root priors for exploration (training only)."""
    actions = list(priors)
    noise = rng.dirichlet([alpha] * len(actions))
    return {
        a: (1 - epsilon) * priors[a] + epsilon * n
        for a, n in zip(actions, noise)
    }


def run_mcts(
    board: Board,
    network: PolicyValueNetwork,
    *,
    n_simulations: int = MCTS_SIMULATIONS,
    c_puct: float = MCTS_C_PUCT,
    add_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> Node:
    """Run MCTS from ``board`` and return the (expanded) root node.

    The root's children carry the visit counts used to choose a move.
    """
    root = Node(prior=0.0)
    legal = generate_legal_moves(board)
    if not legal:
        return root  # terminal position, no search possible

    priors, _ = _evaluate(network, board, legal)
    if add_noise:
        rng = rng or np.random.default_rng()
        priors = _add_dirichlet_noise(
            priors, MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_EPSILON, rng
        )
    for action, prior in priors.items():
        root.children[action] = Node(prior)

    for _ in range(n_simulations):
        _simulate(root, board, network, c_puct)
    return root


def _simulate(
    root: Node, root_board: Board, network: PolicyValueNetwork, c_puct: float
) -> None:
    node = root
    board = root_board.clone()
    path = [root]

    # Selection: descend until we reach an unexpanded node.
    while node.is_expanded():
        action, node = _select_child(node, c_puct)
        board.apply_move(*action_space.index_to_move(action))
        path.append(node)

    # Evaluation / expansion at the leaf.
    legal = generate_legal_moves(board)
    if not legal:
        value = -1.0  # side to move has no reply → it has lost
    else:
        priors, value = _evaluate(network, board, legal)
        for action, prior in priors.items():
            node.children[action] = Node(prior)

    _backpropagate(path, value)


def visit_count_policy(root: Node) -> dict[int, int]:
    """Map of action index → visit count for the root's children."""
    return {action: child.visit_count for action, child in root.children.items()}


__all__ = ["Node", "run_mcts", "visit_count_policy"]
