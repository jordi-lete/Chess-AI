import chess
import chess.pgn

def board_to_array(board):
    piece_map = board.piece_map()
    array = [[0 for _ in range(8)] for _ in range(8)]
    piece_encoding = {
        None: 0,
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }

    # TODO: This convention might be an issue later when integrating with the C++ board
    # There we use file, rank convention with a1 = [0][7]
    for square, piece in piece_map.items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        value = piece_encoding[piece.piece_type]
        if piece.color == chess.BLACK:
            value += 6  # Offset for black pieces
        array[row][col] = value

    return array

def extract_moves_from_pgn(pgn_path, my_color='white'):
    games_data = []

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            player_color = game.headers['White' if my_color == 'white' else 'Black']
            node = game

            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                if board.turn == (my_color == 'white'):
                    board_state = board_to_array(board)
                    move_uci = move.uci()
                    games_data.append((board_state, move_uci))
                board.push(move)
                node = next_node

    return games_data
