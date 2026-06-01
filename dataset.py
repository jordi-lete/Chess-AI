from torch.utils.data import Dataset

class ChessDataset(Dataset):

    def __init__(self, X, y_policy, y_value):
        self.X        = X
        self.y_policy = y_policy
        self.y_value  = y_value   # float tensor, shape (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_policy[idx], self.y_value[idx]