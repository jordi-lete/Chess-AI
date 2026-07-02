import chess
import chess.pgn
import chess.engine
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from chess_utils import board_to_tensor

''' --- Train notebook --- '''
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

''' --- Predict notebook --- '''
def sample_positions_by_phase(pgn_path, n_per_phase=100):
    """Sample positions tagged by rough game phase (opening/middle/endgame)."""
    phases = {'opening': [], 'middlegame': [], 'endgame': []}
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while any(len(v) < n_per_phase for v in phases.values()):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < 10:
                continue

            board = game.board()
            for i, move in enumerate(moves):
                ply = i
                if ply < 10 and len(phases['opening']) < n_per_phase:
                    phases['opening'].append(board.copy())
                elif 10 <= ply < 40 and len(phases['middlegame']) < n_per_phase:
                    if random.random() < 0.1:   # subsample, don't take every ply
                        phases['middlegame'].append(board.copy())
                elif ply >= 40 and len(phases['endgame']) < n_per_phase:
                    if random.random() < 0.2:
                        phases['endgame'].append(board.copy())
                board.push(move)
    return phases


def compare_value_head_to_stockfish(model, device, phases, stockfish_path, depth=12):
    """For each position, get model value and Stockfish eval; compare."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    model.eval()
    results = {}

    try:
        for phase_name, boards in phases.items():
            model_vals, sf_vals = [], []
            for board in boards:
                if board.is_game_over(claim_draw=True):
                    continue

                tensor = torch.tensor(board_to_tensor(board), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, value = model(tensor)
                model_val = float(value.item())

                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                cp = info["score"].relative.score(mate_score=1000)
                sf_val = float(np.tanh(cp / 400.0))

                model_vals.append(model_val)
                sf_vals.append(sf_val)

            model_vals = np.array(model_vals)
            sf_vals = np.array(sf_vals)
            mse = np.mean((model_vals - sf_vals) ** 2)
            sign_agree = np.mean((model_vals > 0) == (sf_vals > 0))
            corr = np.corrcoef(model_vals, sf_vals)[0, 1] if len(model_vals) > 1 else float('nan')

            results[phase_name] = {
                'model_vals': model_vals, 'sf_vals': sf_vals,
                'mse': mse, 'sign_agreement': sign_agree, 'correlation': corr
            }
            print(f"[{phase_name}] n={len(model_vals)}  MSE={mse:.4f}  "
                  f"sign_agreement={sign_agree:.1%}  correlation={corr:.3f}")
    finally:
        engine.quit()

    return results


import matplotlib.pyplot as plt

def plot_value_comparison(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    for ax, (phase_name, r) in zip(axes, results.items()):
        ax.scatter(r['sf_vals'], r['model_vals'], alpha=0.4, s=15)
        ax.plot([-1, 1], [-1, 1], 'r--', linewidth=1, label='perfect agreement')
        ax.set_xlabel('Stockfish eval (tanh-scaled)')
        ax.set_ylabel('Model value head')
        ax.set_title(f"{phase_name}\ncorr={r['correlation']:.2f}  MSE={r['mse']:.3f}")
        ax.legend(fontsize=8)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()