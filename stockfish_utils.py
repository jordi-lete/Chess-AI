import chess
import chess.engine
from mcts import SimpleMCTS
from chess_utils import move_to_policy_index, policy_index_to_move, board_to_tensor
from RL_utils import get_game_result
import random
import torch
import numpy as np

STOCKFISH_ELO = 1320

def play_game_vs_stockfish(model, device, stockfish_path, 
                            stockfish_elo=1320, num_simulations=200,
                            rl_is_white=True, start_position=None,
                            max_moves=200):
    """
    Play one game between the RL model (with MCTS) and Stockfish.
    skill_level: 0-20 (0=~800, 5=~1100, 10=~1500, 15=~1900, 20=full strength)
    Returns (game_history, result) in same format as self-play.
    """
    board = start_position.copy() if start_position else chess.Board()
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": stockfish_elo
    })
    
    mcts = SimpleMCTS(model, device, num_simulations, use_noise=False)
    mcts.reset()
    
    game_history = []
    move_count = 0

    try:
        while not board.is_game_over(claim_draw=True) and move_count < max_moves:
            is_rl_turn = (board.turn == chess.WHITE) == rl_is_white

            if is_rl_turn:
                temperature = 0.0

                action_probs, _ = mcts.get_action_probs(board, temperature)
                if not action_probs:
                    break

                # Build policy vector and store training example
                mcts_policy = np.zeros(4672, dtype=np.float32)
                for move, prob in action_probs.items():
                    mcts_policy[move_to_policy_index(move)] = prob

                board_tensor = torch.tensor(
                    board_to_tensor(board), dtype=torch.float32
                ).unsqueeze(0)
                game_history.append((board_tensor, mcts_policy, board.turn))

                # Select move
                if temperature == 0.0:
                    move = max(action_probs, key=action_probs.get)
                else:
                    probs = mcts_policy / mcts_policy.sum()
                    move_idx = np.random.choice(len(probs), p=probs)
                    move = policy_index_to_move(move_idx, board)
                    if move not in board.legal_moves:
                        move = random.choice(list(board.legal_moves))
            else:
                # Stockfish move — time limit keeps it responsive
                result = engine.play(board, chess.engine.Limit(time=0.05))
                move = result.move

            board.push(move)
            mcts.advance_root(move)
            move_count += 1

    finally:
        engine.quit()

    game_result = get_game_result(board)
    return game_history, game_result


def generate_stockfish_games(model, device, stockfish_path,
                              start_positions, num_games=30,
                              stockfish_elo=None, num_simulations=100,
                              metrics=None):
    data = []
    wins, losses, draws = 0, 0, 0
    
    if stockfish_elo is None:
        stockfish_elo = STOCKFISH_ELO

    for game_idx in range(num_games):
        rl_is_white = (game_idx % 2 == 0)
        start = random.choice(start_positions)

        try:
            game_history, result = play_game_vs_stockfish(
                model, device, stockfish_path,
                stockfish_elo=stockfish_elo,
                num_simulations=num_simulations,
                rl_is_white=rl_is_white,
                start_position=start
            )
        except Exception as e:
            print(f"Stockfish game {game_idx+1} failed: {e}")
            continue

        data.append((game_history, result))

        rl_won = (result > 0 and rl_is_white) or (result < 0 and not rl_is_white)
        if result == 0:
            draws += 1
        elif rl_won:
            wins += 1
        else:
            losses += 1

        print(f"  [Stockfish game {game_idx+1}/{num_games}] "
                 f"Stockfish ELO={stockfish_elo} | "
                 f"RL={'White' if rl_is_white else 'Black'} | "
                 f"result={result:+.0f} | "
                 f"rl_moves={len(game_history)}")

        if metrics is not None:
            rl_perspective = result if rl_is_white else -result   # +1 = model wins
            metrics.sf_results.append(rl_perspective)
            metrics.sf_game_lengths.append(len(game_history))
            if result != 0:
                metrics.sf_terminations['checkmate'] += 1
            else:
                metrics.sf_terminations['draw'] += 1

    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
    print(f"Stockfish batch — ELO={stockfish_elo} | "
             f"W={wins} L={losses} D={draws} | "
             f"score={wins + 0.5*draws:.1f}/{total} ({win_rate:.0%})")

    stats = {'wins': wins, 'losses': losses, 'draws': draws, 'win_rate': win_rate}
    return data, stats

def update_stockfish_elo(current_elo, win_rate,
                          promote_threshold=0.60,
                          demote_threshold=0.20,
                          step=50, min_elo=1320, max_elo=2000):
    if win_rate > promote_threshold and current_elo < max_elo:
        new_elo = min(current_elo + step, max_elo)
        print(f"Stockfish Elo promoted: {current_elo} → {new_elo} "
                 f"(win_rate={win_rate:.0%})")
        return new_elo
    if win_rate < demote_threshold and current_elo > min_elo:
        new_elo = max(current_elo - step, min_elo)
        print(f"Stockfish Elo demoted: {current_elo} → {new_elo} "
                 f"(win_rate={win_rate:.0%})")
        return new_elo
    return current_elo


def play_vs_stockfish_raw(model, device, stockfish_path,
                          num_games=40, stockfish_elo=1320, max_moves=200):
    """
    Raw-policy-only benchmark (no MCTS) — fast, cheap canary metric.
    Independent of MCTS, so it catches policy degradation that the
    MCTS-based training-time Stockfish games might mask.
    """
    import chess_utils

    model.eval()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    results = {"model_win": 0, "draw": 0, "stockfish_win": 0}

    for game_num in range(num_games):
        board = chess.Board()
        model_is_white = (game_num % 2 == 0)
        move_count = 0

        while not board.is_game_over(claim_draw=True) and move_count < max_moves:
            if (board.turn == chess.WHITE) == model_is_white:
                move = chess_utils.test_single_position(model, board, device, debug=False)
                if move not in board.legal_moves:
                    move = list(board.legal_moves)[0]
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            board.push(move)
            move_count += 1

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            results["draw"] += 1
        elif (outcome.winner == chess.WHITE) == model_is_white:
            results["model_win"] += 1
        else:
            results["stockfish_win"] += 1

    engine.quit()
    total = results["model_win"] + results["draw"] + results["stockfish_win"]
    score = (results["model_win"] + 0.5 * results["draw"]) / total if total else 0.0
    print(f"Raw-policy batch — ELO={stockfish_elo} | "
          f"W={results['model_win']} L={results['stockfish_win']} D={results['draw']} | "
          f"score={score:.0%}")
    results['score'] = score
    return results