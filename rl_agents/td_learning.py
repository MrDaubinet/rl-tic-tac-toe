import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import json

class TDAgent:
    def __init__(self, learning_rate: float = 0.2, discount_factor: float = 0.8, epsilon: float = 0.2, player: str = 'O'):
        self.learning_rate = learning_rate  # α (alpha) - learning rate
        self.discount_factor = discount_factor  # γ (gamma) - discount factor
        self.epsilon = epsilon  # ε (epsilon) - for ε-greedy exploration
        self.value_function: Dict[str, float] = {}  # V(s) - state value function
        self.episode_history: List[Tuple[str, int, float]] = []  # [(state, action, reward), ...]
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'
        
    def get_state_key(self, board: List[str]) -> str:
        """Convert board state to string key for value function dictionary."""
        return ''.join(board)
    
    def get_valid_moves(self, board: List[str]) -> List[int]:
        """Get list of empty positions (valid moves)."""
        return [i for i, mark in enumerate(board) if mark == '-']
    
    def get_value(self, state_key: str) -> float:
        """Get value of a state, initialize if not seen before."""
        if state_key not in self.value_function:
            # Initialize with small random value
            self.value_function[state_key] = np.random.uniform(-0.1, 0.1)
        return self.value_function[state_key]
    
    def choose_action(self, board: List[str]) -> int:
        """Choose action using ε-greedy policy."""
        valid_moves = self.get_valid_moves(board)
        
        # Explore: random move
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Exploit: choose best move
        best_value = float('-inf')
        best_moves = []  # Keep track of all moves with the best value
        
        for move in valid_moves:
            # Create copy of board and apply move
            next_board = board.copy()
            next_board[move] = self.player  # Use agent's symbol
            next_state_key = self.get_state_key(next_board)
            value = self.get_value(next_state_key)
            
            # Consider opponent's best response
            min_opponent_value = float('inf')
            for opp_move in self.get_valid_moves(next_board):
                opp_board = next_board.copy()
                opp_board[opp_move] = self.opponent  # Use opponent's symbol
                opp_state_key = self.get_state_key(opp_board)
                opp_value = self.get_value(opp_state_key)
                min_opponent_value = min(min_opponent_value, opp_value)
            
            # Combine immediate value with opponent's response
            combined_value = value - self.discount_factor * min_opponent_value
            
            if combined_value > best_value:
                best_value = combined_value
                best_moves = [move]
            elif combined_value == best_value:
                best_moves.append(move)
        
        # Randomly choose among the best moves
        return random.choice(best_moves)
    
    def update(self, state: str, next_state: str, reward: float):
        """Update value function using TD(0) learning."""
        current_value = self.get_value(state)
        next_value = self.get_value(next_state)
        
        # Scale rewards to make learning more pronounced
        # scaled_reward = reward * 2.0  # Amplify the reward signal
        
        # V(s) ← V(s) + α[r + γV(s') - V(s)]
        td_error = reward + self.discount_factor * next_value - current_value
        new_value = current_value + self.learning_rate * td_error
        
        # Clip values to reasonable range
        self.value_function[state] = np.clip(new_value, -1.0, 1.0)
        
        # Store update magnitude for debugging
        self._last_update = abs(self.learning_rate * td_error)
    
    def reset_episode(self):
        """Reset episode history for new game."""
        self.episode_history = []
    
    def add_to_history(self, state: str, action: int, reward: float):
        """Add state-action-reward tuple to episode history."""
        self.episode_history.append((state, action, reward))
    
    @property
    def last_update(self) -> float:
        """Get the magnitude of the last value function update."""
        return getattr(self, '_last_update', 0.0)

class TDAgentAdvance(TDAgent):
    def __init__(self, learning_rate: float = 0.2, discount_factor: float = 0.8, epsilon: float = 0.2,
                 min_epsilon: float = 0.01, epsilon_decay: float = 0.995,
                 min_learning_rate: float = 0.01, learning_rate_decay: float = 0.995,
                 player: str = 'O'):
        super().__init__(learning_rate, discount_factor, epsilon, player)
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self):
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)

    def on_episode_end(self):
        self.decay_epsilon()
        self.decay_learning_rate()

class TDAgentExporter:
    @staticmethod
    def export_to_json(agent, filename):
        with open(filename, 'w') as f:
            json.dump(agent.value_function, f) 