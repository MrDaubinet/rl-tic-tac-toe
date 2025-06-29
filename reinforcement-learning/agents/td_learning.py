import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import json
import os

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
        """Convert board state to string key with role prefix for role-specific learning."""
        board_str = ''.join(board)
        if hasattr(self, 'role') and self.role is not None:
            return f"{self.role}:{board_str}"
        return board_str
        
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

class RandomAgent(Agent):
    """Agent that makes random valid moves."""
    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """Choose a random valid move."""
        valid_moves = self.get_valid_moves(board)
        return random.choice(valid_moves) if valid_moves else -1

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

class LookAheadAgent(TDAgent):
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
        
        Args:
            board: Current board state
            depth: Remaining search depth
            is_maximizing: True if it's our turn (maximizing player)
            alpha, beta: Alpha-beta pruning bounds
            
        Returns:
            Best evaluation score for this position
        """
        # Check for terminal state
        terminal_value = self.evaluate_terminal_state(board)
        if terminal_value is not None:
            return terminal_value
        
        # If we've reached max depth, use learned value function
        if depth == 0:
            state_key = self.get_state_key(board)
            base_value = self.get_value(state_key)
            # Value function is always from our agent's perspective
            # When it's opponent's turn (not maximizing), we want the negative
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

class LookAheadWithActionValueFunction(LookAheadAgent):
    """
    Q-Learning agent with minimax lookahead that maintains action-value function Q(s,a).
    Combines the strategic depth of minimax with the learning power of Q-values.
    """
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9, 
                 epsilon: float = 0.5, lookahead_depth: int = 2):
        super().__init__(learning_rate, discount_factor, epsilon, lookahead_depth)
        self.action_value_function: Dict[Tuple[str, int], float] = {}
        self.last_state_action = None  # Track (state, action) for updates

    def get_action_value(self, state_key: str, action: int) -> float:
        """Get Q-value for state-action pair. Initialize optimistically if not seen."""
        key = (state_key, action)
        if key not in self.action_value_function:
            self.action_value_function[key] = 0.1  # Optimistic initialization
        return self.action_value_function[key]

    def get_max_action_value(self, state_key: str, valid_actions: List[int]) -> float:
        """Get the maximum Q-value for a state across all valid actions."""
        if not valid_actions:
            return 0.0
        return max(self.get_action_value(state_key, action) for action in valid_actions)

    def minimax_with_qvalues(self, board: List[str], depth: int, is_maximizing: bool, 
                            alpha: float = float('-inf'), beta: float = float('inf')) -> float:
        """
        Enhanced minimax that uses Q-values at leaf nodes instead of state values.
        """
        # Check for terminal state
        terminal_value = self.evaluate_terminal_state(board)
        if terminal_value is not None:
            return terminal_value

        # If we've reached max depth, use learned Q-values
        if depth == 0:
            state_key = self.get_state_key(board)
            valid_moves = self.get_valid_moves(board)
            
            if is_maximizing:
                # Our turn - return max Q-value
                return self.get_max_action_value(state_key, valid_moves)
            else:
                # Opponent's turn - return negative of their best Q-value
                # We assume opponent also tries to maximize their Q-values
                return -self.get_max_action_value(state_key, valid_moves)

        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return 0.0

        if is_maximizing:
            # Our turn - maximize value
            max_eval = float('-inf')
            current_role = self.role
            
            for move in valid_moves:
                new_board = board.copy()
                new_board[move] = current_role
                
                eval_score = self.minimax_with_qvalues(new_board, depth - 1, False, alpha, beta)
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
                
                eval_score = self.minimax_with_qvalues(new_board, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return min_eval

    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """
        Choose action using combination of Q-values and minimax lookahead.
        """
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
            
        # Epsilon-greedy exploration (except in evaluation mode)
        if not evaluation_mode and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        state_key = self.get_state_key(board)
        best_value = float('-inf')
        best_moves = []
        
        # Evaluate each possible move using minimax + Q-values
        for move in valid_moves:
            # Make the move
            new_board = board.copy()
            new_board[move] = self.role
            
            # Combine Q-value with minimax lookahead
            q_value = self.get_action_value(state_key, move)
            
            # Use minimax to evaluate the resulting position
            minimax_value = self.minimax_with_qvalues(new_board, self.lookahead_depth - 1, False)
            
            # Weighted combination of Q-value and minimax (favor minimax for deeper search)
            combined_value = 0.3 * q_value + 0.7 * minimax_value
            
            if combined_value > best_value:
                best_value = combined_value
                best_moves = [move]
            elif combined_value == best_value:
                best_moves.append(move)
        
        return random.choice(best_moves)

    def update(self, state_key: str, next_state_key: str, reward: float):
        """
        Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        # We need the action that was taken to get from state to next_state
        # This should be stored from the last choose_action call
        if self.last_state_action is None:
            return  # Can't update without knowing the action
            
        last_state, last_action = self.last_state_action
        if last_state != state_key:
            return  # State mismatch
            
        current_q_value = self.get_action_value(state_key, last_action)
        
        # For terminal states, next Q-value is just the reward
        if reward != 0:  # Terminal state
            next_max_q = reward
        else:  # Non-terminal state
            next_valid_moves = self.get_valid_moves(self._board_from_state_key(next_state_key))
            next_max_q = self.get_max_action_value(next_state_key, next_valid_moves)
        
        # Q-learning update
        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q_value
        )
        
        self.action_value_function[(state_key, last_action)] = np.clip(new_q_value, -1.0, 1.0)
        
        # Clear the last state-action for next update
        self.last_state_action = None

    def _board_from_state_key(self, state_key: str) -> List[str]:
        """Convert state key back to board representation."""
        # Remove role prefix if present
        if ':' in state_key:
            board_str = state_key.split(':', 1)[1]
        else:
            board_str = state_key
        return list(board_str)

    def add_to_history(self, state: str, action: int, reward: float):
        """Override to track state-action pairs for Q-learning."""
        super().add_to_history(state, action, reward)
        self.last_state_action = (state, action)

    def end_of_episode_update(self, final_reward: float):
        """
        Update Q-value for terminal state-action pair.
        """
        if self.last_state_action is not None:
            state, action = self.last_state_action
            # For terminal states, Q(s,a) = reward
            self.action_value_function[(state, action)] = final_reward
            self.last_state_action = None

    def save(self, directory: str = 'weights', filename: str = None):
        """Save both Q-values and any remaining state values."""
        import os
        import json
        
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f'{self.__class__.__name__}_{self.role}.json'
        
        # Convert tuple keys to strings for JSON serialization
        q_values_serializable = {f"{state}|{action}": value 
                               for (state, action), value in self.action_value_function.items()}
        
        save_data = {
            'action_value_function': q_values_serializable,
            'value_function': self.value_function  # Keep any state values too
        }
        
        path = os.path.join(directory, filename)
        with open(path, 'w') as f:
            json.dump(save_data, f)