import numpy as np
from typing import List
import random
import json

from agents.agent import Agent

class TDAgent(Agent):
    """TD-Learning agent for Tic-Tac-Toe."""
    def get_value(self, state_key: str) -> float:
        """Get the value of a state. Initialize optimistically if not seen before."""
        if state_key not in self.value_function:
            self.value_function[state_key] = 0.1
        return self.value_function[state_key]
        
    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """Choose an action using epsilon-greedy strategy."""
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
            
        if not evaluation_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)
            
        # Choose the action that leads to the highest value
        best_value = float('-inf')
        best_moves = []
        
        for move in valid_moves:
            next_board = board.copy()
            next_board[move] = self.role
            next_state_key = self.get_state_key(next_board)
            value = self.get_value(next_state_key)
            
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
                
        return random.choice(best_moves)
        
    def update(self, state_key: str, next_state_key: str, reward: float):
        """Update value function using TD learning."""
        current_value = self.get_value(state_key)
        
        # For terminal states, the value should be the reward itself
        if reward != 0:  # Terminal state (win/loss/draw)
            next_value = reward
        else:  # Non-terminal state
            next_value = self.get_value(next_state_key)
        
        # TD update
        new_value = current_value + self.learning_rate * (
            reward + self.discount_factor * next_value - current_value
        )
        self.value_function[state_key] = np.clip(new_value, -1.0, 1.0)
        
        # If this is a terminal transition, also set the terminal state value
        if reward != 0:
            self.value_function[next_state_key] = reward
        
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