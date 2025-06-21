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
        

