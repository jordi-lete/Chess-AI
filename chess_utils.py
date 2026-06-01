import chess
import chess.pgn
import numpy as np
import torch
# from model import ChessModel

POLICY_SIZE = 4672  # 4096 normal moves + 192*3 underpromotions

''' ------------------- DATA PREPROCESSING ------------------- '''

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Tensor representation for chess positions.
    
    Args:
        board: chess.Board object
        my_color: True for white's perspective, False for black's
    
    Returns:
        numpy array of shape (18, 8, 8) with the following channels:
        0-5: White pieces (P, N, B, R, Q, K)
        6-11: Black pieces (P, N, B, R, Q, K)
        12: Castling rights (white kingside)
        13: Castling rights (white queenside)
        14: Castling rights (black kingside)
        15: Castling rights (black queenside)
        16: En passant target squares
        17: Check indicator (1.0 if current player is in check)
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Always encode from current player's perspective
    # Current player's pieces in channels 0-5, opponent in 6-11
    my_color = board.turn
    opp_color = not board.turn
    
    piece_to_channel = {
        (chess.PAWN,   my_color):  0,
        (chess.KNIGHT, my_color):  1,
        (chess.BISHOP, my_color):  2,
        (chess.ROOK,   my_color):  3,
        (chess.QUEEN,  my_color):  4,
        (chess.KING,   my_color):  5,
        (chess.PAWN,   opp_color): 6,
        (chess.KNIGHT, opp_color): 7,
        (chess.BISHOP, opp_color): 8,
        (chess.ROOK,   opp_color): 9,
        (chess.QUEEN,  opp_color): 10,
        (chess.KING,   opp_color): 11,
    }
    
    # TODO: This convention might be an issue later when integrating with the C++ board
    # There we use file, rank convention with a1 = [0][7]
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        channel = piece_to_channel[(piece.piece_type, piece.color)]
        tensor[channel, row, col] = 1.0
    
    # Castling rights (channels 12-15)
    if board.turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):  tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):  tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):  tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):  tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[15, :, :] = 1.0
    
    # En passant (channel 16)
    if board.ep_square is not None:
        ep_row = chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        tensor[16, ep_row, ep_col] = 1.0
    
    # Check indicator (channel 17)
    if board.is_check():
        tensor[17, :, :] = 1.0
    
    return tensor


''' ------------------- EXTRACT TRAINING DATA ------------------- '''

def extract_training_data(pgn_path: str, my_color: str = 'white') -> list:
    """
    Extract training data from personal PGN file.
    Returns list of (position_tensor, move_index, value_target).
    value_target: +1.0 = win, 0.0 = draw, -1.0 = loss (from my_color's perspective).
    """
    training_data = []
    my_color_bool = (my_color.lower() == 'white')

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result == "1-0":
                white_value = 1.0
            elif result == "0-1":
                white_value = -1.0
            elif result == "1/2-1/2":
                white_value = 0.0
            else:
                continue  # skip unfinished games

            my_value = white_value if my_color_bool else -white_value

            board = game.board()
            node  = game

            while node.variations:
                next_node = node.variation(0)
                move = next_node.move

                if board.turn == my_color_bool:
                    position_tensor = board_to_tensor(board)
                    move_index = move_to_policy_index(move)
                    training_data.append((position_tensor, move_index, my_value))

                board.push(move)
                node = next_node

    return training_data

def extract_lichess_data(pgn_path: str,
                         min_rating: int = 1600,
                         max_rating: int = 1900,
                         max_games: int = None) -> list:
    """
    Extract training data from a Lichess monthly PGN dump.
    Filters to games where both players are within [min_rating, max_rating].
    Collects positions for both colours.
    Returns list of (position_tensor, move_index, value_target).
    """
    training_data = []
    games_processed = 0

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            if max_games is not None and games_processed >= max_games:
                break

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # --- Rating filter ---
            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
            except ValueError:
                continue

            if not (min_rating <= white_elo <= max_rating
                    and min_rating <= black_elo <= max_rating):
                continue

            result = game.headers.get("Result", "*")
            if result == "1-0":
                white_value = 1.0
            elif result == "0-1":
                white_value = -1.0
            elif result == "1/2-1/2":
                white_value = 0.0
            else:
                continue

            board = game.board()
            node  = game

            while node.variations:
                next_node = node.variation(0)
                move = next_node.move

                # value from the perspective of whoever is to move
                current_value = white_value if board.turn == chess.WHITE else -white_value
                position_tensor = board_to_tensor(board)
                move_index = move_to_policy_index(move)
                training_data.append((position_tensor, move_index, current_value))

                board.push(move)
                node = next_node

            games_processed += 1

    return training_data

def extract_puzzle_data(puzzle_csv_path: str,
                        min_rating: int = None,
                        max_rating: int = None,
                        max_puzzles: int = None) -> list:
    """
    Extract training data from the Lichess puzzle CSV database.
    Download from: https://database.lichess.org/#puzzles

    Each puzzle contributes one position per solution move.
    value_target is always 1.0 — the solver is definitively winning.
    Returns list of (position_tensor, move_index, value_target).
    """
    import csv
    training_data = []
    puzzles_used  = 0

    with open(puzzle_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if max_puzzles is not None and puzzles_used >= max_puzzles:
                break

            # --- Rating filter ---
            if min_rating is not None or max_rating is not None:
                try:
                    rating = int(row['Rating'])
                    if min_rating is not None and rating < min_rating:
                        continue
                    if max_rating is not None and rating > max_rating:
                        continue
                except (ValueError, KeyError):
                    continue

            try:
                board = chess.Board(row['FEN'])
                moves = row['Moves'].split()
            except Exception:
                continue

            if not moves:
                continue

            # Apply the opponent's move that leads into the puzzle
            try:
                board.push_uci(moves[0])
            except Exception:
                continue

            # Remaining moves are the solution
            valid = True
            solver_color = board.turn  # solver moves first after the lead-in move

            for move_uci in moves[1:]:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        valid = False
                        break

                    # Only store the solver's moves, not forced opponent replies
                    if board.turn == solver_color:
                        position_tensor = board_to_tensor(board)
                        move_index = move_to_policy_index(move)
                        training_data.append((position_tensor, move_index, 1.0))

                    board.push(move)
                except Exception:
                    valid = False
                    break

            if valid:
                puzzles_used += 1

    return training_data

''' ------------------- TRAINING ------------------- '''

def _promo_offset_idx(from_square: int, to_square: int) -> int:
    """
    Map a promotion (from, to) pair to a direction index 0/1/2.
    0 = captures toward a-file, 1 = straight push, 2 = captures toward h-file.
    Works regardless of colour.
    """
    return chess.square_file(to_square) - chess.square_file(from_square) + 1

def move_to_policy_index(move: chess.Move) -> int:
    """
    Convert a chess move to a policy index.

    Layout:
      0    - 4095 : normal moves and queen promotions (from*64 + to)
      4096 - 4287 : knight underpromotions (from*3 + offset)
      4288 - 4479 : bishop underpromotions
      4480 - 4671 : rook underpromotions
    """
    from_sq = move.from_square
    to_sq   = move.to_square

    if move.promotion is None or move.promotion == chess.QUEEN:
        # Queen promotions share the normal-move index — the board context
        # (pawn on 7th rank) disambiguates during inference.
        return from_sq * 64 + to_sq

    offset_idx = _promo_offset_idx(from_sq, to_sq)  # 0, 1, or 2

    if move.promotion == chess.KNIGHT:
        return 4096 + from_sq * 3 + offset_idx
    elif move.promotion == chess.BISHOP:
        return 4288 + from_sq * 3 + offset_idx
    else:  # ROOK
        return 4480 + from_sq * 3 + offset_idx


def create_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Create a mask for legal moves — updated for POLICY_SIZE = 4672."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.float32)
    for move in board.legal_moves:
        mask[move_to_policy_index(move)] = 1.0
    return mask

''' ------------------- PREDICTING ------------------- '''

def policy_index_to_move(policy_index: int, board: chess.Board) -> chess.Move:
    """Convert a policy index back to a chess.Move."""

    if policy_index < 4096:
        from_sq = policy_index // 64
        to_sq   = policy_index % 64
        # Detect queen promotion from board state
        promotion = None
        piece = board.piece_at(from_sq)
        if (piece and piece.piece_type == chess.PAWN
                and chess.square_rank(to_sq) in (0, 7)):
            promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion)

    # --- Underpromotion ---
    if policy_index < 4288:
        promo_type = chess.KNIGHT
        idx = policy_index - 4096
    elif policy_index < 4480:
        promo_type = chess.BISHOP
        idx = policy_index - 4288
    else:
        promo_type = chess.ROOK
        idx = policy_index - 4480

    from_sq    = idx // 3
    offset_idx = idx % 3           # 0 = left-file capture, 1 = straight, 2 = right-file capture

    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file   = from_file + offset_idx - 1
    to_rank   = 7 if from_rank == 6 else 0   # white promotes to rank 7, black to rank 0

    to_sq = chess.square(to_file, to_rank)
    return chess.Move(from_sq, to_sq, promo_type)

def test_single_position(model, board, device):
    best_move = None

    # Convert board to tensor
    position_tensor = board_to_tensor(board)
    input_tensor = torch.tensor(position_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(input_tensor)

    # Build legal move mask
    legal_mask = torch.full_like(policy_logits, -1e9)  # -inf for illegal
    for move in board.legal_moves:
        idx = move_to_policy_index(move)
        legal_mask[0, idx] = 0  # unmask legal moves

    masked_logits = policy_logits + legal_mask
    probabilities = torch.softmax(masked_logits, dim=1)

    # Get top 5 moves among legal ones
    top_moves = torch.topk(probabilities, 5)

    print("Top 5 predicted legal moves:")
    for i, (prob, move_idx) in enumerate(zip(top_moves.values[0], top_moves.indices[0])):
        move = policy_index_to_move(move_idx.item(), board)
        print(f"{i+1}. {move} (confidence: {prob:.1%})")
        if best_move is None:
            best_move = move

    return best_move