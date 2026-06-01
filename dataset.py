import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    """Lazy dataset — converts to tensors on demand, avoiding a ~19GB upfront allocation."""
    def __init__(self, data):
        self.data = data  # list of (np.ndarray, int, float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_tensor, move_idx, value = self.data[idx]
        return (
            torch.tensor(board_tensor, dtype=torch.float32),
            torch.tensor(move_idx,     dtype=torch.long),
            torch.tensor(value,        dtype=torch.float32),
        )