import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """
    Stores self-play games as (game_history, result) where:
      - game_history is a list of (board_tensor, policy_vector, turn)
        board_tensor can be a torch.Tensor (CPU), policy_vector is np.ndarray
      - result is a scalar (WHITE's POV): +1 white win, -1 black win, 0 draw/other
    """

    def __init__(self, capacity=10000, rng=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = rng if rng is not None else random.Random()

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def add_games(self, games):
        """
        games: iterable of (game_history, result)
          where game_history: list of (board_tensor, policy_vector, turn)
        We store CPU-safe copies to avoid GPU memory leaks.
        """
        for game_history, result in games:
            safe_history = []
            for board_tensor, policy_vector, turn in game_history:
                # board_tensor could be torch.Tensor on GPU or CPU; move to CPU
                if isinstance(board_tensor, torch.Tensor):
                    bt_cpu = board_tensor.detach().cpu()
                else:
                    bt_cpu = torch.tensor(board_tensor, dtype=torch.float32)

                # policy_vector to numpy float32
                if isinstance(policy_vector, np.ndarray):
                    pv = policy_vector.astype(np.float32)
                else:
                    pv = np.array(policy_vector, dtype=np.float32)

                safe_history.append((bt_cpu, pv, turn))
            # store tuple (safe_history, result) - keeps same external format as self_play_data
            self.buffer.append((safe_history, float(result)))

    def sample_examples(self, n_examples):
        """
        Sample n_examples uniformly across positions inside games.
        Returns list of (board_tensor, policy_vector, value) where:
          - board_tensor is a torch.Tensor (CPU) (shape [1,C,H,W])
          - policy_vector is np.ndarray (4288,)
          - value is float from perspective of player to move in that stored position (+1 = player to move winning)
        """
        if len(self.buffer) == 0:
            return []

        examples = []
        for _ in range(n_examples):
            game_idx = self.rng.randrange(len(self.buffer))
            game_history, result = self.buffer[game_idx]
            if len(game_history) == 0:
                continue
            pos_idx = self.rng.randrange(len(game_history))
            board_tensor, policy_vector, turn = game_history[pos_idx]

            # convert result (white POV) to current player's perspective
            value = result if turn else -result

            # Return board_tensor as CPU torch.Tensor (no .to(device) here)
            examples.append((board_tensor, policy_vector, value))

        return examples

    def get_all_examples(self):
        """Flatten all stored games as examples (may be big)."""
        ex = []
        for game_history, result in self.buffer:
            for board_tensor, policy_vector, turn in game_history:
                value = result if turn else -result
                ex.append((board_tensor, policy_vector, value))
        return ex