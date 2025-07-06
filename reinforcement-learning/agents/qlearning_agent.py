from typing import Dict, List, Tuple
import random
import numpy as np
from agents.agent import Agent

class QLearning(Agent):
    """
    Pure Q-Learning agent that maintains action-value function Q(s,a).
    Learns the value of each action in each state through experience.
    """
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9, 
                 epsilon: float = 0.5, lookahead_depth: int = 2):
        super().__init__(learning_rate, discount_factor, epsilon)
        self.action_value_function: Dict[Tuple[str, int], float] = {}
        self.previous_state_action = None  # Track (state, action) for Q-learning updates
        
    def reset_episode(self):
        """Reset episode-specific tracking."""
        super().reset_episode()
        self.previous_state_action = None

    def get_action_value(self, state_key: str, action: int) -> float:
        """Get Q-value for state-action pair. Initialize optimistically."""
        key = (state_key, action)
        if key not in self.action_value_function:
            self.action_value_function[key] = 0.1  # Optimistic initialization
        return self.action_value_function[key]

    def get_max_action_value(self, state_key: str, valid_actions: List[int]) -> float:
        """Get the maximum Q-value for a state across all valid actions."""
        if not valid_actions:
            return 0.0
        return max(self.get_action_value(state_key, action) for action in valid_actions)

    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """
        Choose action using pure Q-learning (epsilon-greedy).
        Store the state-action pair for future Q-learning updates.
        """
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
            
        state_key = self.get_state_key(board)
        
        # Epsilon-greedy exploration (except in evaluation mode)
        if not evaluation_mode and random.random() < self.epsilon:
            action = random.choice(valid_moves)
        else:
            # Choose action with highest Q-value
            best_value = float('-inf')
            best_moves = []
            
            for move in valid_moves:
                q_value = self.get_action_value(state_key, move)
                
                if q_value > best_value:
                    best_value = q_value
                    best_moves = [move]
                elif q_value == best_value:
                    best_moves.append(move)
            
            action = random.choice(best_moves)
        
        # Store state-action for Q-learning update
        self.previous_state_action = (state_key, action)
        return action

    def update(self, state_key: str, next_state_key: str, reward: float):
        """
        Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        Updates the Q-value for the PREVIOUS state-action pair that led to current state.
        """
        if self.previous_state_action is None:
            return  # No previous action to update
            
        prev_state, prev_action = self.previous_state_action
        current_q_value = self.get_action_value(prev_state, prev_action)
        
        # Calculate target Q-value
        if reward != 0:  # Terminal state - no future rewards
            target_q = reward
        else:  # Non-terminal state - add discounted future value
            next_valid_moves = self.get_valid_moves(self._board_from_state_key(next_state_key))
            next_max_q = self.get_max_action_value(next_state_key, next_valid_moves)
            target_q = reward + self.discount_factor * next_max_q
        
        # Q-learning update
        new_q_value = current_q_value + self.learning_rate * (target_q - current_q_value)
        self.action_value_function[(prev_state, prev_action)] = np.clip(new_q_value, -1.0, 1.0)

    def _board_from_state_key(self, state_key: str) -> List[str]:
        """Convert state key back to board representation."""
        # Remove role prefix if present
        if ':' in state_key:
            board_str = state_key.split(':', 1)[1]
        else:
            board_str = state_key
        return list(board_str)

    def end_of_episode_update(self, final_reward: float):
        """
        Final Q-value update for the last action taken in the episode.
        """
        if self.previous_state_action is not None:
            state, action = self.previous_state_action
            current_q = self.get_action_value(state, action)
            # For terminal states, target Q-value is just the final reward
            new_q = current_q + self.learning_rate * (final_reward - current_q)
            self.action_value_function[(state, action)] = np.clip(new_q, -1.0, 1.0)
            self.previous_state_action = None

    def save(self, directory: str = 'weights', filename: str = None):
        """Save Q-values as a nested dictionary: {state: {action: value}}."""
        import os
        import json
        
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f'{self.__class__.__name__}_{self.role}.json'
        
        # Convert to nested dict: {state: {action: value}}
        q_values_nested = {}
        for (state, action), value in self.action_value_function.items():
            if state not in q_values_nested:
                q_values_nested[state] = {}
            q_values_nested[state][str(action)] = value
        
        path = os.path.join(directory, filename)
        with open(path, 'w') as f:
            json.dump(q_values_nested, f)

    def load(self, directory: str = 'weights', filename: str = None):
        """Load Q-values."""
        import os
        import json
        
        if filename is None:
            filename = f'{self.__class__.__name__}_{self.role}.json'
        
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                q_values_data = json.load(f)
            # Convert string keys back to tuples
            self.action_value_function = {}
            for key, value in q_values_data.items():
                if '|' in key:
                    state, action = key.split('|', 1)
                    self.action_value_function[(state, int(action))] = value