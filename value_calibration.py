import chess
import chess.pgn
import chess.engine
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from chess_utils import board_to_tensor


def sample_positions_for_annotation(pgn_path, num_positions=20000,
                                    min_ply=6, max_ply=80):
    """Sample diverse positions from a PGN file as chess.Board objects."""
    boards = []
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while len(boards) < num_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            moves = list(game.mainline_moves())
            if len(moves) < min_ply:
                continue

            usable_range = range(min_ply, min(len(moves), max_ply))
            if not usable_range:
                continue
            n_samples = min(3, len(usable_range))
            sample_plies = sorted(random.sample(list(usable_range), n_samples))

            board = game.board()
            ply_idx = 0
            for i, move in enumerate(moves):
                if ply_idx < len(sample_plies) and i == sample_plies[ply_idx]:
                    boards.append(board.copy())
                    ply_idx += 1
                board.push(move)

    return boards[:num_positions]


def annotate_positions_with_stockfish(boards, stockfish_path, depth=10):
    """
    Run Stockfish on each position. Returns list of (position_tensor, value)
    where value is normalised to [-1, 1] from the CURRENT PLAYER's
    perspective — matching board_to_tensor's convention.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    data = []
    try:
        for i, board in enumerate(boards):
            if board.is_game_over(claim_draw=True):
                continue

            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].relative   # already current-player POV
            cp = score.score(mate_score=1000)

            value = float(np.tanh(cp / 400.0))   # squash centipawns to [-1, 1]
            data.append((board_to_tensor(board), value))

            if (i + 1) % 1000 == 0:
                print(f"Annotated {i+1}/{len(boards)} positions")
    finally:
        engine.quit()

    return data


class ValueDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor, value = self.data[idx]
        return (torch.tensor(tensor, dtype=torch.float32),
                torch.tensor(value, dtype=torch.float32))


def finetune_value_head(model, value_data, device,
                        epochs=8, lr=1e-4, batch_size=256, val_fraction=0.1):
    """Fine-tune ONLY the value head against Stockfish-eval labels."""
    for name, param in model.named_parameters():
        param.requires_grad = 'value_head' in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable (value head only): {trainable:,} params")

    dataset = ValueDataset(value_data)
    val_size = int(val_fraction * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    loss_fn  = torch.nn.MSELoss()
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        for name, module in model.named_modules():
            if any(name.startswith(p) for p in ('conv_in', 'bn_in', 'res_blocks')):
                module.eval()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            _, value_pred = model(inputs)
            loss = loss_fn(value_pred.squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, sign_correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, value_pred = model(inputs)
                val_loss += loss_fn(value_pred.squeeze(-1), targets).item()
                sign_correct += ((value_pred.squeeze(-1) > 0) == (targets > 0)).sum().item()
                n_val += targets.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader)
        sign_acc  = sign_correct / n_val

        marker = " ✓" if avg_val < best_val else ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'models/value_calibrated_best.pth')

        print(f"Epoch {epoch+1}/{epochs}: train MSE {avg_train:.4f}  "
              f"val MSE {avg_val:.4f}  sign_acc {sign_acc:.1%}{marker}")

    model.load_state_dict(torch.load('models/value_calibrated_best.pth'))
    for p in model.parameters():
        p.requires_grad = True
    return model