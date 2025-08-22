import torch
from model import ChessModel
from ResNet_model import ResNetChessModel

''' ------------------ LOAD MODEL ------------------ '''

def load_model(model_path=None, input_channels=19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = ChessModel(input_channels=input_channels).to(device)
    
    # Load saved weights
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    else:
        print("Initialized new model.")
    
    model.eval()  # Set to evaluation mode
    return model, device

def load_resnet_model(model_path=None, input_channels=19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = ResNetChessModel(input_channels=input_channels).to(device)

    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    else:
        print("Initialized new model.")

    model.train()
    return model, device
        

''' ------------------ TRAINING DATA ------------------ '''

import chess
import chess.pgn
from chess_utils import board_to_tensor, create_legal_moves_mask
import numpy as np
import random
import math

def extract_middlegame_positions(pgn_path: str, num_positions=1000):
    """
    Extract middle game positions from PGN files
    
    Args:
        pgn_path: Path to PGN file
        num_positions: Number of positions to extract
        min_move: Minimum move number to consider
        max_move: Maximum move number to consider
    
    Returns:
        List of chess.Board objects representing middle game positions
    """
    positions = []
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while len(positions) < num_positions:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            board = game.board()
            node = game
            move_count = 0

            # generate random number of moves to play
            num_moves_to_play = math.floor(16 + 20 * random.random()) & ~1
            
            # Play through the game
            while node.variations and len(positions) < num_positions and move_count < num_moves_to_play:
                next_node = node.variation(0)
                move = next_node.move
                board.push(move)
                move_count += 1
                node = next_node
            
            # Additional filters for middle game characteristics
            piece_count = len(board.piece_map())
            if piece_count >= 12 and piece_count <= 28:  # Not too few pieces (endgame) or too many (opening)
                # eval_score = evaluator.evaluate_position(board)
                # if -0.5 <= eval_score <= 0.5:  # Only add posisions that are roughly equal
                positions.append(board.copy())

    print(f"Extracted {len(positions)} middle game positions")
    return positions


def extract_varied_positions(pgn_path: str, num_positions=1000, opening_frac=0.2, early_frac=0.3):
    """
    Extract a mix of opening, early-game, and middlegame positions from PGN files.

    Args:
        pgn_path: Path to PGN file
        num_positions: Total number of positions to extract
        opening_frac: Fraction of positions from opening (move 0-4)
        early_frac: Fraction from early-game (move 5-12)
        The rest will be middlegame (move 13+ with piece count filter)

    Returns:
        List of chess.Board objects
    """
    positions = []
    opening_positions = int(num_positions * opening_frac)
    early_positions = int(num_positions * early_frac)
    middlegame_positions = num_positions - opening_positions - early_positions

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while len(positions) < num_positions:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            node = game
            move_count = 0

            # Decide which type to extract
            if len(positions) < opening_positions:
                # Opening: random move between 0 and 4
                num_moves_to_play = random.randint(0, 4)
            elif len(positions) < opening_positions + early_positions:
                # Early-game: random move between 5 and 12
                num_moves_to_play = random.randint(5, 12)
            else:
                # Middlegame: random move between 13 and 20
                num_moves_to_play = random.randint(13, 20)

            # Play through the game
            while node.variations and move_count < num_moves_to_play:
                next_node = node.variation(0)
                move = next_node.move
                board.push(move)
                move_count += 1
                node = next_node

            # For middlegame, filter by piece count
            if len(positions) >= opening_positions + early_positions:
                piece_count = len(board.piece_map())
                if piece_count < 12 or piece_count > 28:
                    continue  # Skip if not a typical middlegame
            if board.is_game_over():
                continue # Skip if already checkmate

            positions.append(board.copy())

    print(f"Extracted {len(positions)} varied positions (openings, early, middlegame)")
    return positions


''' ------------------ SELF-PLAY GAME LOOP ------------------ '''

from chess_utils import policy_index_to_move, move_to_policy_index
import torch.nn.functional as F

def play_self_play_game(model, device, starting_position, max_moves=100, temperature=1.0):
    """
    Play one self-play game using model policy sampling
    """
    board = starting_position.copy()
    game_history = []

    model.eval()
    
    for move_count in range(max_moves):
        if board.is_game_over():
            break

        # Prepare input
        position_tensor = board_to_tensor(board)
        input_tensor = torch.tensor(position_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        legal_moves_mask = create_legal_moves_mask(board).to(device)

        with torch.no_grad():
            policy_logits, value_pred = model(input_tensor)
            
            # Check for NaN in model output
            if torch.isnan(policy_logits).any() or torch.isnan(value_pred).any():
                print("Warning: NaN detected in model output!")
                break
                
            masked_probs = apply_legal_moves_mask(policy_logits, legal_moves_mask)
        
        # Sample move
        move = select_move_with_exploration(masked_probs, board, temperature)
        move_idx = move_to_policy_index(move)

        # Store state and data
        game_history.append((
            input_tensor.cpu(),                   # board state
            legal_moves_mask.cpu(),              # legal moves mask
            move_idx,                            # selected move index
            board.turn                           # who played: True=white, False=black
        ))

        board.push(move)

    # Determine outcome
    final_result = get_game_result(board)
    return game_history, final_result

from mcts import SimpleMCTS

def play_game_with_mcts(model, device, starting_board, max_moves=100, temperature=1.0, num_simulations=400):
    """Play a game using MCTS"""
    board = starting_board.copy()
    model.eval()
    mcts = SimpleMCTS(model, device, num_simulations)
    game_history = []
    
    while not board.is_game_over() and len(game_history) < max_moves:
        temperature = max(0.1, 1.0 * (0.95 ** len(game_history)))
        # temperature = 1.0 if len(game_history) < 10 else 0.1
        # Get MCTS action probabilities
        action_probs, root = mcts.get_action_probs(board, temperature)
        
        # Store training data
        board_tensor = board_to_tensor(board)
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Convert action probs to policy vector
        policy_vector = np.zeros(4288)
        for move, prob in action_probs.items():
            move_idx = move_to_policy_index(move)
            policy_vector[move_idx] = prob
        
        game_history.append((board_tensor, policy_vector, board.turn))
        
        # Select and make move
        moves = list(action_probs.keys())
        probs = list(action_probs.values())
        move = np.random.choice(moves, p=probs)
        board.push(move)

    # Get final result
    result = get_game_result(board)
    print(f"Game finished: {result}")
    return game_history, result

def get_game_result(board):
    """Convert chess game outcome to RL reward"""
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1  # Winner gets +1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_repetition() or board.can_claim_draw():
        return 0  # Draw
    else:
        return 0  # Unfinished game treated as draw
    
def apply_legal_moves_mask(move_logits, legal_moves_mask):
    """Apply legal moves mask and renormalize probabilities"""
    # Zero out illegal moves (legal move mask is 1 if legal 0 if illegal)
    masked_logits = move_logits + (legal_moves_mask - 1) * 1e9

    # Convert to probabilities
    probabilities = F.softmax(masked_logits, dim=1)
    
    return probabilities

def select_move_with_exploration(masked_probs, board, temperature=1.0):
    """Select move with some exploration noise"""
    if temperature > 0:
        # Add exploration by sampling from probability distribution
        probs = masked_probs.squeeze().cpu().numpy()
        probs = np.power(probs, 1/temperature)
        probs = probs / np.sum(probs)
        
        move_idx = np.random.choice(len(probs), p=probs)
    else:
        # Greedy selection (no exploration)
        move_idx = torch.argmax(masked_probs).item()
    
    return policy_index_to_move(move_idx, board)

''' --------------- RL model ----------------'''

def compute_loss(policy_logits, value_pred, move_targets, value_targets, legal_mask):
    # Mask illegal moves
    masked_logits = policy_logits + (legal_mask - 1) * 1e9
    policy_loss = F.cross_entropy(masked_logits, move_targets)

    value_loss = F.mse_loss(value_pred.squeeze(), value_targets.float())

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss

def compute_loss_mcts(policy_logits, value_preds, policy_targets, value_targets, kl_div, epoch):
    kl_beta = max(0.05, 0.5 * (0.95 ** epoch))
    # Policy loss: cross-entropy between predicted logits and soft targets
    policy_log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.sum(policy_targets * policy_log_probs, dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_preds.view(-1), value_targets)

    total_loss = policy_loss + value_loss + kl_beta*kl_div
    return total_loss, policy_loss, value_loss

def blend_policies(mcts_policy, cnn_policy, alpha=0.7):
    blended = alpha * cnn_policy + (1 - alpha) * mcts_policy
    blended /= blended.sum()  # Normalize
    return blended