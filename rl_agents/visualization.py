import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path

class LearningVisualizer:
    """A class for visualizing and tracking learning metrics in reinforcement learning."""
    
    def __init__(self):
        """Initialize the learning visualizer with empty tracking data."""
        self.episodes: List[int] = []
        self.win_rates_O: List[float] = []
        self.win_rates_X: List[float] = []
        self.rewards_O: List[float] = []
        self.rewards_X: List[float] = []
        self.value_functions: Dict[str, np.ndarray] = {}
        
        # Set up plotting style
        plt.style.use('default')  # Use default style instead of seaborn
        self.colors = {
            'win_rate_O': '#2ecc71',
            'win_rate_X': '#e67e22',
            'reward_O': '#3498db',
            'reward_X': '#e74c3c',
            'value_positive': '#e74c3c',
            'value_negative': '#3498db',
            'value_neutral': '#95a5a6'
        }
    
    def update_win_rates(self, episode: int, win_rate_O: float, win_rate_X: float) -> None:
        """
        Update the win rate tracking data for both O and X agents.
        
        Args:
            episode (int): Current episode number
            win_rate_O (float): Win rate value for O agent between 0 and 1
            win_rate_X (float): Win rate value for X agent between 0 and 1
        """
        self.episodes.append(episode)
        self.win_rates_O.append(win_rate_O)
        self.win_rates_X.append(win_rate_X)
    
    def update_rewards_both(self, episode: int, reward_O: float, reward_X: float) -> None:
        """
        Update the reward tracking data for both O and X agents.
        
        Args:
            episode (int): Current episode number
            reward_O (float): Reward value for O agent
            reward_X (float): Reward value for X agent
        """
        if len(self.episodes) == 0 or self.episodes[-1] != episode:
            self.episodes.append(episode)
        self.rewards_O.append(reward_O)
        self.rewards_X.append(reward_X)
    
    def update_value_function(self, board_state: List[int], value_function: np.ndarray) -> None:
        """
        Update the value function data for a given board state.
        
        Args:
            board_state (List[int]): Current board state as a flat list
            value_function (np.ndarray): Value function array for the board state
        """
        state_key = ''.join(map(str, board_state))
        self.value_functions[state_key] = value_function
    
    def plot_learning_curves(self, 
                           save_path: Optional[str] = None,
                           show_win_rate: bool = True,
                           show_rewards: bool = False,
                           window_size: int = 100) -> None:
        """
        Plot learning curves showing win rates and rewards over episodes.
        Args:
            save_path (str, optional): Path to save the plot
            show_win_rate (bool): Whether to show win rate curves
            show_rewards (bool): Whether to show rewards curves
            window_size (int): (Unused, kept for compatibility)
        """
        plt.figure(figsize=(12, 6))
        if show_win_rate and self.win_rates_O and self.win_rates_X:
            plt.plot(self.episodes, self.win_rates_O, color=self.colors['win_rate_O'], label='Win Rate O')
            plt.plot(self.episodes, self.win_rates_X, color=self.colors['win_rate_X'], label='Win Rate X')
        if show_rewards and self.rewards_O and self.rewards_X:
            plt.plot(self.episodes, self.rewards_O, color=self.colors['reward_O'], label='Reward O')
            plt.plot(self.episodes, self.rewards_X, color=self.colors['reward_X'], label='Reward X')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.title('Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_value_function_heatmap(self,
                                  board_state: List[int],
                                  save_path: Optional[str] = None) -> None:
        """
        Plot a heatmap of the value function for a given board state.
        
        Args:
            board_state (List[int]): Current board state as a flat list
            save_path (str, optional): Path to save the plot
        """
        state_key = ''.join(map(str, board_state))
        if state_key not in self.value_functions:
            raise ValueError("No value function data available for this board state")
        
        values = self.value_functions[state_key].reshape(3, 3)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(values,
                   annot=True,
                   fmt='.2f',
                   cmap='RdBu',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Value'})
        
        plt.title('Value Function Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.episodes.clear()
        self.win_rates_O.clear()
        self.win_rates_X.clear()
        self.rewards_O.clear()
        self.rewards_X.clear()
        self.value_functions.clear()
    
    def save_data(self, save_dir: str) -> None:
        """
        Save tracking data to files.
        
        Args:
            save_dir (str): Directory to save the data
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save win rates and rewards
        if self.win_rates_O or self.win_rates_X:
            np.save(save_path / 'win_rates_O.npy',
                   np.array([self.episodes, self.win_rates_O]))
            np.save(save_path / 'win_rates_X.npy',
                   np.array([self.episodes, self.win_rates_X]))
        if self.rewards_O or self.rewards_X:
            np.save(save_path / 'rewards_O.npy',
                   np.array([self.episodes, self.rewards_O]))
            np.save(save_path / 'rewards_X.npy',
                   np.array([self.episodes, self.rewards_X]))
        
        # Save value functions
        if self.value_functions:
            # Convert value functions to a format suitable for numpy
            value_data = {
                state: values.tolist() 
                for state, values in self.value_functions.items()
            }
            np.save(save_path / 'value_functions.npy',
                   np.array([value_data], dtype=object))
    
    def load_data(self, load_dir: str) -> None:
        """
        Load tracking data from files.
        
        Args:
            load_dir (str): Directory to load the data from
        """
        load_path = Path(load_dir)
        
        # Load win rates and rewards
        if (load_path / 'win_rates_O.npy').exists():
            data = np.load(load_path / 'win_rates_O.npy')
            self.episodes = data[0].tolist()
            self.win_rates_O = data[1].tolist()
        
        if (load_path / 'win_rates_X.npy').exists():
            data = np.load(load_path / 'win_rates_X.npy')
            self.win_rates_X = data[1].tolist()
        
        if (load_path / 'rewards_O.npy').exists():
            data = np.load(load_path / 'rewards_O.npy')
            self.rewards_O = data[1].tolist()
        
        if (load_path / 'rewards_X.npy').exists():
            data = np.load(load_path / 'rewards_X.npy')
            self.rewards_X = data[1].tolist()
        
        # Load value functions
        if (load_path / 'value_functions.npy').exists():
            data = np.load(load_path / 'value_functions.npy', allow_pickle=True)
            value_data = data[0]
            self.value_functions = {
                state: np.array(values)
                for state, values in value_data.items()
            }
    
    @staticmethod
    def _moving_average(values: List[float], window: int) -> np.ndarray:
        """
        Calculate the moving average of a list of values.
        
        Args:
            values (List[float]): List of values to average
            window (int): Window size for the moving average
        
        Returns:
            np.ndarray: Smoothed values
        """
        weights = np.ones(window) / window
        return np.convolve(values, weights, mode='valid')
    
    def plot_policy_heatmap(self, agent, board_state: list, save_path: Optional[str] = None) -> None:
        """
        Plot a heatmap of the agent's value for each possible move in a given board state, highlighting the best move.
        Args:
            agent: The TD-learning agent
            board_state (list): The board state to visualize (list of 9 chars)
            save_path (str, optional): Path to save the plot
        """
        values = np.full((3, 3), np.nan)
        best_value = float('-inf')
        best_move = None
        for i in range(9):
            if board_state[i] == '-':
                next_board = board_state.copy()
                next_board[i] = 'O'
                state_key = agent.get_state_key(next_board)
                v = agent.get_value(state_key)
                values[i // 3, i % 3] = v
                if v > best_value:
                    best_value = v
                    best_move = i
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(values, annot=True, fmt='.2f', cmap='RdBu', center=0, square=True, cbar_kws={'label': 'Value'})
        if best_move is not None:
            y, x = divmod(best_move, 3)
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='gold', lw=3))
        plt.title('Policy Heatmap (Best move outlined)')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_value_distribution(self, agent, save_path: Optional[str] = None) -> None:
        """
        Plot a histogram of all state values in the agent's value function.
        Args:
            agent: The TD-learning agent
            save_path (str, optional): Path to save the plot
        """
        values = list(agent.value_function.values())
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30, color=self.colors['value_positive'], alpha=0.7)
        plt.xlabel('State Value')
        plt.ylabel('Count')
        plt.title('Distribution of State Values')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 