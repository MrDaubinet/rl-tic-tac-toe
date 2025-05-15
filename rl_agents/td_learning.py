import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import json
import os

class TDAgent:
    def __init__(self, learning_rate: float = 0.2, discount_factor: float = 0.8, epsilon: float = 0.2, player: str = 'O', debug: bool = False):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.value_function: Dict[str, float] = {}
        self.episode_history: List[Tuple[str, int, float]] = []
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'
        self.debug = debug
        self.last_state = None  # Track the last state for terminal updates

    def get_state_key(self, board: List[str]) -> str:
        return ''.join(board)

    def get_valid_moves(self, board: List[str]) -> List[int]:
        """Get list of valid moves. Returns empty list if game is already won."""
        # First check if game is already won
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        # Check for any winning line
        for line in lines:
            if all(board[i] == 'X' for i in line) or all(board[i] == 'O' for i in line):
                return []  # Game is already won, no valid moves
        
        # If game isn't won, return empty spaces
        return [i for i, mark in enumerate(board) if mark == '-']

    def get_value(self, state_key: str, debug: bool = False) -> float:
        """Get the value of a state. Initialize optimistically if not seen before."""
        if state_key not in self.value_function:
            if debug:
                print("initializing value function for state", state_key, "to 0.5")
            # Optimistic initialization to encourage exploration
            self.value_function[state_key] = 0
        return self.value_function[state_key]

    def choose_action(self, board: List[str]) -> int:
        """
        Choose an action based on the value function.

        Returns -1 if no valid moves exist.
        """
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:  # Handle case where no valid moves exist
            return -1
            
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        best_value = float('-inf')
        best_moves = []
        for move in valid_moves:
            next_board = board.copy()
            next_board[move] = self.player
            next_state_key = self.get_state_key(next_board)
            value = self.get_value(next_state_key)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        return random.choice(best_moves)

    def update(self, state: str, next_state: str, reward: float):
        """
        Update the value function for a given state transition.
        For non-terminal transitions, update using TD learning.
        Terminal state updates are handled in end_of_episode_update.
        """
        current_value = self.get_value(state)
        next_value = self.get_value(next_state)
            
        # Update the current state's value
        td_error = reward + self.discount_factor * next_value - current_value
        new_value = current_value + self.learning_rate * td_error
        self.value_function[state] = np.clip(new_value, -1.0, 1.0)
        self._last_update = abs(self.learning_rate * td_error)
        
        # Track the last state for terminal updates
        self.last_state = next_state

    def reset_episode(self):
        """Reset episode-specific variables."""
        self.last_state = None
        self.episode_history = []

    def add_to_history(self, state: str, action: int, reward: float):
        self.episode_history.append((state, action, reward))

    @property
    def last_update(self) -> float:
        return getattr(self, '_last_update', 0.0)

    def save(self, directory: str = 'weights', filename: str = None):
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f'{self.__class__.__name__}_{self.player}.json'
        path = os.path.join(directory, filename)
        with open(path, 'w') as f:
            json.dump(self.value_function, f)

    def decay_epsilon(self, decay_rate: float = 0.9999, min_epsilon: float = 0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def end_of_episode_update(self, final_reward: float):
        """
        Handle terminal state updates at the end of an episode.
        For terminal states, we set their value to the final reward.
        """
        if self.last_state is not None:
            # For terminal states, set value directly to reward
            self.value_function[self.last_state] = final_reward
            if self.debug:
                print("updating value function for terminal state", self.last_state, final_reward)
            self.last_state = None  # Reset for next episode

class LookAheadAgent(TDAgent):
    def choose_action(self, board: List[str]) -> int:
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
            
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
            
        best_value = float('-inf')
        best_moves = []
        
        for move in valid_moves:
            next_board = board.copy()
            next_board[move] = self.player
            
            # If not winning, evaluate considering opponent's response
            next_state_key = self.get_state_key(next_board)
            value = self.get_value(next_state_key)
            min_opponent_value = float('inf')
            
            for opp_move in self.get_valid_moves(next_board):
                opp_board = next_board.copy()
                opp_board[opp_move] = self.opponent
                opp_state_key = self.get_state_key(opp_board)
                opp_value = self.get_value(opp_state_key)
                min_opponent_value = min(min_opponent_value, opp_value)
            
            combined_value = value - self.discount_factor * min_opponent_value
            if combined_value > best_value:
                best_value = combined_value
                best_moves = [move]
            elif combined_value == best_value:
                best_moves.append(move)
        
        return random.choice(best_moves)

class MonteCarloAgent(TDAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_history = []

    def add_to_history(self, state: str, action: int, reward: float):
        self.episode_history.append((state, action, reward))

    def reset_episode(self):
        self.episode_history = []

    def update(self, state: str, next_state: str, reward: float):
        # Do nothing; MC updates at end of episode
        pass

    def end_of_episode_update(self, final_reward: float):
        # Update all visited states with the final reward
        for state, action, reward in self.episode_history:
            current_value = self.get_value(state)
            td_error = final_reward - current_value
            new_value = current_value + self.learning_rate * td_error
            self.value_function[state] = np.clip(new_value, -1.0, 1.0) 