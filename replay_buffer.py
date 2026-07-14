import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """
    Stores games as (game_history, result) where:
      game_history: list of (board_tensor, policy_vector, value)
        - board_tensor: torch.Tensor CPU, shape [1,C,H,W]
        - policy_vector: np.ndarray [4672]
        - value: float, current-player perspective [-1, 1]
          (Stockfish eval for SF games; game outcome for any self-play)
      result: scalar game outcome from White's POV (+1/-1/0)
    """

    def __init__(self, capacity=5000, rng=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = rng if rng is not None else random.Random()

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def add_games(self, games):
        for game_history, result in games:
            safe_history = []
            for board_tensor, policy_vector, value in game_history:
                # board_tensor to CPU
                if isinstance(board_tensor, torch.Tensor):
                    bt_cpu = board_tensor.detach().cpu()
                else:
                    bt_cpu = torch.tensor(board_tensor, dtype=torch.float32)

                # policy_vector to numpy float32
                if isinstance(policy_vector, np.ndarray):
                    pv = policy_vector.astype(np.float32)
                else:
                    pv = np.array(policy_vector, dtype=np.float32)

                # value is already a float — just ensure correct type
                v = float(value)

                safe_history.append((bt_cpu, pv, v))

            self.buffer.append((safe_history, float(result)))

    def sample_examples(self, n_examples):
        """
        Sample n_examples uniformly across positions.
        Returns list of (board_tensor, policy_vector, value) where
        value is already in current-player perspective — no conversion needed.
        """
        if len(self.buffer) == 0:
            return []

        game_lengths = [len(gh) for gh, _ in self.buffer]
        total_positions = sum(game_lengths)
        if total_positions == 0:
            return []
        weights = [l / total_positions for l in game_lengths]

        examples = []
        for _ in range(n_examples):
            game_idx = self.rng.choices(range(len(self.buffer)), weights=weights)[0]
            game_history, result = self.buffer[game_idx]
            if not game_history:
                continue
            pos_idx = self.rng.randrange(len(game_history))
            board_tensor, policy_vector, value = game_history[pos_idx]
            examples.append((board_tensor, policy_vector, value))

        return examples

    def get_all_examples(self):
        ex = []
        for game_history, result in self.buffer:
            for board_tensor, policy_vector, value in game_history:
                ex.append((board_tensor, policy_vector, value))
        return ex