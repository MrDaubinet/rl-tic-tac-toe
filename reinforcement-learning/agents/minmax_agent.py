import random
from typing import List
import numpy as np

from agents.td_agent import TDAgent

class MinMaxTDAgent(TDAgent):
    """
    Minimax-style LookAhead agent that uses proper game tree search.
    Combines learned value function with minimax lookahead for better play.
    """
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9, 
                 epsilon: float = 0.5, lookahead_depth: int = 2):
        super().__init__(learning_rate, discount_factor, epsilon)
        self.lookahead_depth = lookahead_depth
    
    def evaluate_terminal_state(self, board: List[str]) -> float:
        """Evaluate terminal game states"""
        # Check for wins manually using the same logic as environment
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        # Check if we won
        for line in lines:
            if all(board[i] == self.role for i in line):
                return 1.0  # We win
        
        # Check if opponent won
        opponent_role = 'O' if self.role == 'X' else 'X'
        for line in lines:
            if all(board[i] == opponent_role for i in line):
                return -1.0  # Opponent wins
        
        # Check for draw (board full)
        if '-' not in board:
            return 0.0  # Draw
        
        return None  # Not terminal
    
    def minimax(self, board: List[str], depth: int, is_maximizing: bool, 
                alpha: float = float('-inf'), beta: float = float('inf')) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        """
        # Check for terminal state
        terminal_value = self.evaluate_terminal_state(board)
        if terminal_value is not None:
            return terminal_value
        
        # If we've reached max depth, use learned value function
        if depth == 0:
            state_key = self.get_state_key(board)
            base_value = self.get_value(state_key)
            return base_value if is_maximizing else -base_value
        
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return 0.0  # No moves available (shouldn't happen in valid states)
        
        if is_maximizing:
            # Our turn - maximize value
            max_eval = float('-inf')
            current_role = self.role
            
            for move in valid_moves:
                new_board = board.copy()
                new_board[move] = current_role
                
                eval_score = self.minimax(new_board, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return max_eval
        else:
            # Opponent's turn - minimize our value
            min_eval = float('inf')
            opponent_role = 'O' if self.role == 'X' else 'X'
            
            for move in valid_moves:
                new_board = board.copy()
                new_board[move] = opponent_role
                
                eval_score = self.minimax(new_board, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return min_eval
    
    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """
        Choose action using minimax lookahead combined with epsilon-greedy exploration.
        """
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
            
        # Epsilon-greedy exploration (except in evaluation mode)
        if not evaluation_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        best_value = float('-inf')
        best_moves = []
        
        # Evaluate each possible move using minimax
        for move in valid_moves:
            # Make the move
            new_board = board.copy()
            new_board[move] = self.role
            
            # Use minimax to evaluate the resulting position
            # We start with depth-1 since we already made our move
            move_value = self.minimax(new_board, self.lookahead_depth - 1, False)
            
            if move_value > best_value:
                best_value = move_value
                best_moves = [move]
            elif move_value == best_value:
                best_moves.append(move)
        
        return random.choice(best_moves)
    
    def update(self, state_key: str, next_state_key: str, reward: float):
        """
        Clean TD learning update without the flawed lookahead corruption.
        The minimax lookahead is only used for action selection, not learning.
        """
        current_value = self.get_value(state_key)
        
        # For terminal states, the value should be the reward itself
        if reward != 0:  # Terminal state (win/loss/draw)
            next_value = reward
        else:  # Non-terminal state
            next_value = self.get_value(next_state_key)
        
        # Standard TD update - clean and simple
        new_value = current_value + self.learning_rate * (
            reward + self.discount_factor * next_value - current_value
        )
        self.value_function[state_key] = np.clip(new_value, -1.0, 1.0)
        
        # If this is a terminal transition, also set the terminal state value
        if reward != 0:
            self.value_function[next_state_key] = reward