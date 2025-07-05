import json
import os
from typing import Dict, List, Tuple

class Agent:
    """Base class for all agents."""
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9, epsilon: float = 0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.value_function: Dict[str, float] = {}
        self.episode_history: List[Tuple[str, int, float]] = []  # Changed from history to episode_history
        self.role = None  # 'X' or 'O', assigned at episode start
        self.last_state = None  # Track the last state for terminal updates
        
    def set_role(self, role: str):
        """Set the agent's role (X or O) for the current episode."""
        self.role = role
        
    def get_valid_moves(self, board: List[str]) -> List[int]:
        """Get list of empty positions."""
        return [i for i, mark in enumerate(board) if mark == '-']
        
    def get_state_key(self, board: List[str]) -> str:
        """Convert board state to string key (no role prefix, since weights are per-role)."""
        return ''.join(board)
        
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.episode_history = []
        self.last_state = None
        
    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """Choose an action given the current board state."""
        raise NotImplementedError
        
    def update(self, state_key: str, next_state_key: str, reward: float):
        """Update value function for a state transition."""
        pass
        
    def add_to_history(self, state: str, action: int, reward: float):
        """Add a state-action-reward tuple to history."""
        self.episode_history.append((state, action, reward))
        self.last_state = state
        
    def end_of_episode_update(self, final_reward: float):
        """Update value function at end of episode using TD learning for terminal state only."""
        if not self.episode_history:
            return
            
        # Only update the final state with the terminal reward
        # The intermediate states have already been updated via TD learning
        if self.last_state:
            current_value = self.get_value(self.last_state)
            self.value_function[self.last_state] = current_value + self.learning_rate * (
                final_reward - current_value
            )
        
    def decay_epsilon(self, decay_rate: float = 0.9999, min_epsilon: float = 0.01):
        """Decay exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        
    def save(self, directory: str = 'weights', filename: str = None):
        """Save the agent's value function to a file."""
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f'{self.__class__.__name__}_{self.role}.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as f:
            json.dump(self.value_function, f)