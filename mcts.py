import math
import torch
import numpy as np
import chess
from chess_utils import board_to_tensor, move_to_policy_index
from RL_utils import get_game_result
import random
import torch.nn.functional as F

DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.visits = 0
        self.total_value = 0.0
        self.children = {}
        self.expanded = False
        
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def ucb_score(self, c_puct=1.0):
        if self.visits == 0:
            return float('inf')
        
        u_score = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + u_score
    
    def select_child(self, c_puct=1.0):
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))
    
    def expand(self, model, device, add_noise=False):
        if self.expanded or self.board.is_game_over():
            return

        board_tensor = board_to_tensor(self.board)
        input_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = model(input_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        # value is network output in WHITE's POV (tanh -> [-1,1])
        v = float(value.item())
        # convert to the *current node* player's perspective (+ = good for player to move)
        value_for_current_player = v if self.board.turn == chess.WHITE else -v

        # Mask illegal moves
        legal_moves = list(self.board.legal_moves)
        legal_indices = [move_to_policy_index(move) for move in legal_moves]
        mask = np.zeros_like(policy_probs)
        mask[legal_indices] = 1
        policy_probs *= mask
        if policy_probs.sum() > 0:
            policy_probs /= policy_probs.sum()
        else:
            policy_probs = mask / mask.sum()

        # Add Dirichlet noise at root
        if add_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                idx = move_to_policy_index(move)
                policy_probs[idx] = (1 - DIRICHLET_EPSILON) * policy_probs[idx] + DIRICHLET_EPSILON * noise[i]

        legal_mass = policy_probs[legal_indices].sum() if len(legal_indices) > 0 else 0.0
        if legal_mass > 0:
            policy_probs /= legal_mass
        else:
            policy_probs = mask / mask.sum()

        for move in legal_moves:
            move_idx = move_to_policy_index(move)
            prior = policy_probs[move_idx]
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(child_board, parent=self, move=move, prior=prior)

        self.expanded = True
        return value_for_current_player
    
    def backup(self, value):
        self.visits += 1
        self.total_value += value
        
        if self.parent:
            self.parent.backup(-value)  # Flip value for opponent


class SimpleMCTS:
    def __init__(self, model, device, num_simulations=200, c_puct=1.0):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, board):
        root = MCTSNode(board)
        for i in range(self.num_simulations):
            node = root
            while node.expanded and not node.board.is_game_over():
                node = node.select_child(self.c_puct)
            if node.board.is_game_over():
                result = get_game_result(node.board)
                value = result if node.board.turn == chess.WHITE else -result
            else:
                value = node.expand(self.model, self.device, add_noise=(node.parent is None))
                if value is None:
                    continue
            node.backup(value)
        return root
    
    def get_action_probs(self, board, temperature=1.0):
        root = self.search(board)
        if not root.children:
            return {}, None
        visits = {move: child.visits for move, child in root.children.items()}
        total_visits = sum(visits.values())
        moves = list(visits.keys())
        visit_counts = np.array(list(visits.values()), dtype=np.float32)
        if total_visits == 0:
            prob = 1.0 / len(visits)
            action_probs = {move: prob for move in moves}
        else:
            if temperature == 0:
                best_move = moves[np.argmax(visit_counts)]
                action_probs = {move: (1.0 if move == best_move else 0.0) for move in moves}
            else:
                if temperature != 1.0:
                    visit_counts = np.power(visit_counts, 1.0 / temperature)
                probs = visit_counts / np.sum(visit_counts)
                action_probs = {move: prob for move, prob in zip(moves, probs)}
        return action_probs, root
    
    def select_move(self, board, temperature=1.0):
        action_probs, root = self.get_action_probs(board, temperature)
        
        if not action_probs:
            return random.choice(list(board.legal_moves))
        
        moves = list(action_probs.keys())
        probs = list(action_probs.values())
        
        if temperature == 0:
            return max(action_probs, key=action_probs.get)
        else:
            return np.random.choice(moves, p=probs)
        
