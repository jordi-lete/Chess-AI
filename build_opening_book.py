# build_opening_book.py
import chess
import chess.pgn
import json
from collections import defaultdict

def build_opening_book(pgn_paths: list, max_opening_moves: int = 15) -> dict:
    """
    Build a frequency-weighted opening book from PGN files.
    Returns a dict mapping FEN (no move counters) -> {uci_move: count}
    """
    book = defaultdict(lambda: defaultdict(int))

    for pgn_path, my_color in pgn_paths:
        my_color_bool = (my_color.lower() == 'white')
        with open(pgn_path, 'r', encoding='utf-8') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                node = game
                move_num = 0

                while node.variations and move_num < max_opening_moves * 2:
                    next_node = node.variation(0)
                    move = next_node.move

                    # Only record YOUR moves
                    if board.turn == my_color_bool:
                        # Strip move counters from FEN for position matching
                        fen_key = ' '.join(board.fen().split()[:4])
                        book[fen_key][move.uci()] += 1

                    board.push(move)
                    node = next_node
                    move_num += 1

    # Convert to plain dicts for serialisation
    return {fen: dict(moves) for fen, moves in book.items()}


def get_book_move(book: dict, board: chess.Board) -> chess.Move | None:
    """
    Look up the current position in the book.
    Returns the most frequently played move, or None if out of book.
    """
    fen_key = ' '.join(board.fen().split()[:4])
    if fen_key not in book:
        return None

    moves = book[fen_key]
    best_uci = max(moves, key=moves.get)
    return chess.Move.from_uci(best_uci)


# Build and save
book = build_opening_book([
    ("games/jalba20-white.pgn", "white"),
    ("games/jalba20-black.pgn", "black"),
    ("games/Jeedy20-white.pgn", "white"),
    ("games/Jeedy20-black.pgn", "black"),
])

with open("opening_book.json", "w") as f:
    json.dump(book, f)

print(f"Book positions: {len(book):,}")