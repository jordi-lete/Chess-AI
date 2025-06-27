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
                eval_score = evaluator.evaluate_position(board)
                if -0.5 <= eval_score <= 0.5:  # Only add posisions that are roughly equal
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

from chess_utils import policy_index_to_move

def play_self_play_game(model, device, starting_position, max_moves=100):
    """
    Play a single self-play game from a given starting position
    
    Returns:
        - game_history: List of (board_state, move_probs, legal_moves_mask)
        - game_result: 1 for white win, -1 for black win, 0 for draw
    """
    board = starting_position.copy()
    game_history = []
    
    for move_count in range(max_moves):
        if board.is_game_over():
            break
            
        # Get model prediction
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        legal_moves_mask = create_legal_moves_mask(board).to(device)
        
        with torch.no_grad():
            move_probs = model(board_tensor)
            
        # Apply legal moves mask and add exploration noise
        masked_probs = apply_legal_moves_mask(move_probs, legal_moves_mask)
        
        # Select move (with some exploration)
        move = select_move_with_exploration(masked_probs, board)
        
        # Store this position and move probabilities for training
        game_history.append((
            board.copy(),
            masked_probs.cpu(),
            legal_moves_mask.cpu()
        ))
        
        # Make the move
        board.push(move)
    
    # Determine game result
    game_result = get_game_result(board)
    
    return game_history, game_result

def get_game_result(board):
    """Convert chess game outcome to RL reward"""
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1  # Winner gets +1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_repetition():
        return 0  # Draw
    else:
        return 0  # Unfinished game treated as draw
    
def apply_legal_moves_mask(move_probs, legal_moves_mask):
    """Apply legal moves mask and renormalize probabilities"""
    # Zero out illegal moves (legal move mask is 1 if legal 0 if illegal)
    masked_probs = move_probs * legal_moves_mask
    
    # Renormalize so probabilities sum to 1
    masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)
    
    return masked_probs

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