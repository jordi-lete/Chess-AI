import torch
from model import ChessModel

''' ------------------ LOAD MODEL ------------------ '''

def load_model(model_path, input_channels=19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = ChessModel(input_channels=input_channels).to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.train()  # Set to evaluation mode
    return model, device


''' ------------------ STOCKFISH ------------------ '''

from stockfish import Stockfish

class PositionEvaluator:
    def __init__(self, stockfish_path, elo_rating=1400):
        """Initialize Stockfish engine for position evaluation"""
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_elo_rating(elo_rating)
        self.stockfish.update_engine_parameters({"UCI_LimitStrength": True})
        print(f"Stockfish initialized with ELO: {elo_rating}")
    
    def evaluate_position(self, board):
        """
        Get position evaluation in normalized form
        Returns value between -1.0 and 1.0 (from white's perspective)
        """
        try:
            self.stockfish.set_fen_position(board.fen())
            info = self.stockfish.get_evaluation()
            
            if info is None:
                return 0.0
            
            eval_type = info["type"]
            score = info["value"]
            
            if eval_type == "mate":
                # Mate scores: positive for white advantage, negative for black
                return 1.0 if score > 0 else -1.0
            else:
                # Centipawn scores: normalize to [-1, 1] range
                normalized_score = max(-10.0, min(10.0, score / 100.0))
                return normalized_score
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0
    
    def get_best_moves(self, board, num_moves=3):
        """Get top N moves from Stockfish"""
        try:
            self.stockfish.set_fen_position(board.fen())
            return self.stockfish.get_top_moves(num_moves)
        except:
            return []
        

''' ------------------ TRAINING DATA ------------------ '''

import chess
import chess.pgn
from chess_utils import board_to_tensor, create_legal_moves_mask
import numpy as np
import random
import math

def extract_middlegame_positions(pgn_path: str, evaluator: PositionEvaluator, num_positions=1000):
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

def create_training_batch(positions, batch_size=32):
    """
    Create a training batch from positions
    
    Returns:
        - board_tensors: torch.Tensor of shape [batch_size, 19, 8, 8]
        - legal_moves_masks: torch.Tensor of shape [batch_size, 4288]
        - boards: List of chess.Board objects for reference
    """
    
    # Sample random positions
    batch_positions = random.sample(positions, min(batch_size, len(positions)))
    
    # Convert to tensors
    board_tensors = []
    legal_moves_masks = []
    boards = []
    
    for board in batch_positions:
        # Convert board to tensor
        board_tensor = board_to_tensor(board)
        board_tensors.append(board_tensor)
        
        # Create legal moves mask
        legal_mask = create_legal_moves_mask(board)
        legal_moves_masks.append(legal_mask)
        
        boards.append(board.copy())
    
    # Stack into batches
    board_tensors = torch.tensor(np.stack(board_tensors), dtype=torch.float32)
    legal_moves_masks = torch.stack(legal_moves_masks)
    
    return board_tensors, legal_moves_masks, boards



''' ------------------ SELF-PLAY GAME LOOP ------------------ '''

from chess_utils import policy_index_to_move, move_to_policy_index
import torch.nn.functional as F

def self_play_game(model, device, temperature=1.0, max_moves=200):
    board = chess.Board()
    trajectory = []

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        # Convert board to tensor
        board_tensor = board_to_tensor(board)
        input_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            policy_logits, value = model(input_tensor)
            policy_logits = policy_logits[0].cpu()
            value = value.item()

        # Mask illegal moves
        mask = create_legal_moves_mask(board)
        masked_logits = policy_logits + torch.log(mask + 1e-8)  # log(0) becomes -inf

        # Sample move
        probs = torch.softmax(masked_logits / temperature, dim=0)
        move_idx = torch.multinomial(probs, 1).item()
        move = policy_index_to_move(move_idx, board)

        if not board.is_legal(move):
            print("Warning: model chose illegal move.")
            break

        trajectory.append((board_tensor, probs.numpy(), board.turn))  # Save (s, π, player)
        board.push(move)

    # Final game result: +1/-1/0 from white's perspective
    result = board.result()
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0

    # Create final dataset: [(s_t, π_t, z)] with z adjusted for each player
    data = []
    for board_tensor, pi, player_color in trajectory:
        z = outcome if player_color == chess.WHITE else -outcome
        data.append((board_tensor, pi, z))

    return data

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


def update_model(model, optimiser, game_histories, game_results, device):
    """
    Update model using policy gradient method (REINFORCE)
    
    Args:
        model: The neural network model
        optimiser: PyTorch optimiser
        game_histories: List of game histories from self-play
        game_results: List of game results (1, -1, or 0)
        device: torch device
    """
    model.train()  # Set to training mode
    
    total_loss = 0
    num_positions = 0
    
    # Process each game
    for game_history, game_result in zip(game_histories, game_results):
        
        # Process each position in the game
        for board_tensor, legal_moves_mask, actual_move_idx, player_color in game_history:
            board_tensor = board_tensor.to(device)
            legal_moves_mask = legal_moves_mask.to(device)
            
            # Calculate reward for this position
            # Reward is from the perspective of the player who made the move
            if player_color:  # White player
                reward = game_result
            else:  # Black player
                reward = -game_result
            
            # Forward pass
            move_logits = model(board_tensor)
            
            # Apply legal moves mask
            move_probs = apply_legal_moves_mask(move_logits, legal_moves_mask)
            
            # Calculate policy loss (negative log probability weighted by reward)
            log_prob = torch.log(move_probs[0, actual_move_idx] + 1e-8)
            policy_loss = -log_prob * reward
            
            total_loss += policy_loss
            num_positions += 1
    
    # Average the loss
    if num_positions > 0:
        avg_loss = total_loss / num_positions
        
        # Backpropagation
        optimiser.zero_grad()
        avg_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update model parameters in-place
        optimiser.step()
        
        return avg_loss.item()
    
    return 0.0