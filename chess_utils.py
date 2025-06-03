import chess
import chess.pgn
import numpy as np
import torch

def board_to_tensor(board: chess.Board, my_color: bool) -> np.ndarray:
    """
    Tensor representation for chess positions.
    
    Args:
        board: chess.Board object
        my_color: True for white's perspective, False for black's
    
    Returns:
        numpy array of shape (19, 8, 8) with the following channels:
        0-5: White pieces (P, N, B, R, Q, K)
        6-11: Black pieces (P, N, B, R, Q, K)
        12: Castling rights (white kingside)
        13: Castling rights (white queenside)
        14: Castling rights (black kingside)
        15: Castling rights (black queenside)
        16: En passant target squares
        17: Turn indicator (1.0 if it's my_color's turn)
        18: Check indicator (1.0 if current player is in check)
    """
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    
    # Piece channels (0-11)
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    # TODO: This convention might be an issue later when integrating with the C++ board
    # There we use file, rank convention with a1 = [0][7]
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        channel = piece_to_channel[(piece.piece_type, piece.color)]
        tensor[channel, row, col] = 1.0
    
    # Castling rights (channels 12-15)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    
    # En passant (channel 16)
    if board.ep_square is not None:
        ep_row = chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        tensor[16, ep_row, ep_col] = 1.0
    
    # Turn indicator (channel 17)
    if board.turn == my_color:
        tensor[17, :, :] = 1.0
    
    # Check indicator (channel 18)
    if board.is_check():
        tensor[18, :, :] = 1.0
    
    return tensor


def extract_training_data(pgn_path: str, my_color: str = 'white') -> list:
    """
    Extract training data from PGN file.
    
    Returns list of tuples: (position_tensor, move_policy_index)
    """
    training_data = []
    my_color_bool = (my_color.lower() == 'white')
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            board = game.board()
            node = game
            
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                
                # Only collect positions where it's my turn
                if board.turn == my_color_bool:
                    position_tensor = board_to_tensor(board, my_color_bool)
                    move_index = move_to_policy_index(move)
                    training_data.append((position_tensor, move_index))
                
                board.push(move)
                node = next_node
    
    return training_data


def move_to_policy_index(move: chess.Move) -> int:
    """
    Convert a chess move to a policy index for neural network output.
    This uses the standard approach where each square can make moves to 
    any other square, plus promotion moves.
    
    Total policy size: 64 * 64 + 64 * 3 = 4288
    (64*64 for normal moves, 64*3 for underpromotions to N,B,R)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Normal moves: from_square * 64 + to_square
    base_index = from_square * 64 + to_square
    
    # Handle promotions
    if move.promotion:
        if move.promotion == chess.QUEEN:
            return base_index
        elif move.promotion == chess.KNIGHT:
            return 4096 + from_square  # Indices 4096-4159
        elif move.promotion == chess.BISHOP:
            return 4160 + from_square  # Indices 4160-4223
        elif move.promotion == chess.ROOK:
            return 4224 + from_square  # Indices 4224-4287
    
    return base_index


def create_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Create a mask for legal moves in the policy vector."""
    mask = torch.zeros(4288, dtype=torch.float32)
    
    for move in board.legal_moves:
        policy_idx = move_to_policy_index(move)
        mask[policy_idx] = 1.0
    
    return mask


def policy_index_to_move(policy_index: int, board: chess.Board) -> chess.Move:
    """Convert policy index back to chess move."""
    if policy_index < 4096:
        # Normal move
        from_square = policy_index // 64
        to_square = policy_index % 64
        promotion = chess.QUEEN if board.piece_at(from_square) and \
                   board.piece_at(from_square).piece_type == chess.PAWN and \
                   (chess.square_rank(to_square) in [0, 7]) else None
    else:
        # Underpromotion
        if policy_index < 4160:
            from_square = policy_index - 4096
            to_square = from_square + (8 if board.turn == chess.WHITE else -8)
            promotion = chess.KNIGHT
        elif policy_index < 4224:
            from_square = policy_index - 4160
            to_square = from_square + (8 if board.turn == chess.WHITE else -8)
            promotion = chess.BISHOP
        else:
            from_square = policy_index - 4224
            to_square = from_square + (8 if board.turn == chess.WHITE else -8)
            promotion = chess.ROOK
    
    return chess.Move(from_square, to_square, promotion)
