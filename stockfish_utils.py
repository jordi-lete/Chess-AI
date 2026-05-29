import chess
import chess.engine
from mcts import SimpleMCTS
from chess_utils import move_to_policy_index, policy_index_to_move, board_to_tensor
from RL_utils import get_game_result
import random
import torch
import numpy as np

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
    
    mcts = SimpleMCTS(model, device, num_simulations)
    mcts.reset()
    
    game_history = []
    move_count = 0

    try:
        while not board.is_game_over(claim_draw=True) and move_count < max_moves:
            is_rl_turn = (board.turn == chess.WHITE) == rl_is_white

            if is_rl_turn:
                # Actual ply in the real game
                actual_ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
                temperature = 1.0 if actual_ply < 30 else 0.5

                action_probs, _ = mcts.get_action_probs(board, temperature)
                if not action_probs:
                    break

                # Build policy vector and store training example
                mcts_policy = np.zeros(4288, dtype=np.float32)
                for move, prob in action_probs.items():
                    mcts_policy[move_to_policy_index(move)] = prob

                board_tensor = torch.tensor(
                    board_to_tensor(board), dtype=torch.float32
                ).unsqueeze(0)
                game_history.append((board_tensor, mcts_policy, board.turn))

                # Select move
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

