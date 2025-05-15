import numpy as np
from typing import List, Tuple
from rl_agents.td_learning import TDAgentAdvance, TDAgentExporter
from rl_agents.visualization import LearningVisualizer
import os

class TicTacToeEnv:
    """Simple Tic-Tac-Toe environment."""
    
    def __init__(self):
        self.board = ['-'] * 9
        self.current_player = 'X'
    
    def reset(self) -> List[str]:
        """Reset the game board."""
        self.board = ['-'] * 9
        self.current_player = 'X'
        return self.board.copy()
    
    def step(self, action: int) -> Tuple[List[str], float, bool]:
        """
        Make a move on the board.
        
        Args:
            action: Position to place the mark (0-8)
            
        Returns:
            tuple: (new_state, reward, done)
        """
        if self.board[action] != '-':
            return self.board.copy(), -10, True  # Invalid move
        
        self.board[action] = self.current_player
        
        # Check for win
        if self._check_win():
            return self.board.copy(), 1.0, True
        
        # Check for draw
        if '-' not in self.board:
            return self.board.copy(), 0.0, True
        
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self.board.copy(), 0.0, False
    
    def _check_win(self) -> bool:
        """Check if current player has won."""
        # Winning combinations
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for line in lines:
            if all(self.board[i] == self.current_player for i in line):
                return True
        return False

def random_opponent_move(board: List[str]) -> int:
    """Make a random valid move."""
    valid_moves = [i for i, mark in enumerate(board) if mark == '-']
    return np.random.choice(valid_moves) if valid_moves else -1

def train_agent(episodes: int = 10000,
                save_dir: str = 'training_data',
                visualize_every: int = 100) -> None:
    """
    Train two TD-learning agents via self-play and visualize their progress.
    """
    env = TicTacToeEnv()
    agent_O = TDAgentAdvance(learning_rate=0.1, discount_factor=0.9, epsilon=0.2,
                            min_epsilon=0.01, epsilon_decay=0.995,
                            min_learning_rate=0.01, learning_rate_decay=0.995,
                            player='O')
    agent_X = TDAgentAdvance(learning_rate=0.1, discount_factor=0.9, epsilon=0.2,
                            min_epsilon=0.01, epsilon_decay=0.995,
                            min_learning_rate=0.01, learning_rate_decay=0.995,
                            player='X')
    visualizer = LearningVisualizer()
    os.makedirs(save_dir, exist_ok=True)
    wins_O = 0
    wins_X = 0
    draws = 0
    total_games = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        episode_reward_O = 0
        episode_reward_X = 0
        # Randomize starting player
        current_player = np.random.choice(['O', 'X'])
        while not done:
            if current_player == 'O':
                action = agent_O.choose_action(state)
                next_state, reward, done = env.step(action)
                state_key = agent_O.get_state_key(state)
                next_state_key = agent_O.get_state_key(next_state)
                agent_O.update(state_key, next_state_key, reward)
                if done:
                    episode_reward_O += reward
                    episode_reward_X -= reward
                    if reward == 1.0:
                        wins_O += 1
                    elif reward == 0.0:
                        draws += 1
                    break
                state = next_state
                current_player = 'X'
            else:
                action = agent_X.choose_action(state)
                next_state, reward, done = env.step(action)
                state_key = agent_X.get_state_key(state)
                next_state_key = agent_X.get_state_key(next_state)
                agent_X.update(state_key, next_state_key, reward)
                if done:
                    episode_reward_X += reward
                    episode_reward_O -= reward
                    if reward == 1.0:
                        wins_X += 1
                    elif reward == 0.0:
                        draws += 1
                    break
                state = next_state
                current_player = 'O'
        total_games += 1
        # Decay epsilon and learning rate for both agents
        agent_O.on_episode_end()
        agent_X.on_episode_end()
        # Update visualization data for O (can also do for X if desired)
        if episode % visualize_every == 0:
            win_rate_O = wins_O / total_games
            win_rate_X = wins_X / total_games
            visualizer.update_win_rates(episode, win_rate_O, win_rate_X)
            visualizer.update_rewards_both(episode, episode_reward_O, episode_reward_X)
            empty_board = ['-'] * 9
            values = np.zeros((3, 3))
            for i in range(9):
                next_board = empty_board.copy()
                next_board[i] = 'O'
                next_state = agent_O.get_state_key(next_board)
                values[i // 3, i % 3] = agent_O.get_value(next_state)
                max_opponent_value = float('-inf')
                for j in range(9):
                    if j != i:
                        opponent_board = next_board.copy()
                        opponent_board[j] = 'X'
                        opponent_state = agent_O.get_state_key(opponent_board)
                        opponent_value = agent_O.get_value(opponent_state)
                        max_opponent_value = max(max_opponent_value, opponent_value)
                if max_opponent_value != float('-inf'):
                    values[i // 3, i % 3] -= max_opponent_value * agent_O.discount_factor
            if np.max(np.abs(values)) > 0:
                values = values / np.max(np.abs(values))
            visualizer.update_value_function(empty_board, values)
            visualizer.save_data(save_dir)
            visualizer.plot_learning_curves(
                save_path=os.path.join(save_dir, f'learning_curves_{episode}.png')
            )
            visualizer.plot_value_function_heatmap(
                empty_board,
                save_path=os.path.join(save_dir, f'value_heatmap_{episode}.png')
            )
            visualizer.plot_policy_heatmap(
                agent_O,
                empty_board,
                save_path=os.path.join(save_dir, f'policy_heatmap_{episode}.png')
            )
            visualizer.plot_value_distribution(
                agent_O,
                save_path=os.path.join(save_dir, f'value_distribution_{episode}.png')
            )
            print(f"Episode {episode}/{episodes}")
            print(f"O Win Rate: {win_rate_O:.2f} | X Win Rate: {win_rate_X:.2f} | Draws: {draws/total_games:.2f}")
            print(f"Episode Reward (O): {episode_reward_O:.2f} | (X): {episode_reward_X:.2f}")
            print(f"Value function range: [{values.min():.2f}, {values.max():.2f}]")
            print("Saved: learning curves, value heatmap, policy heatmap, value distribution.")
            print("-" * 40)

    # After training, export both agents
    TDAgentExporter.export_to_json(agent_O, 'web-app/static/agent_O.json')
    TDAgentExporter.export_to_json(agent_X, 'web-app/static/agent_X.json')

if __name__ == "__main__":
    train_agent(episodes=50000, visualize_every=1000) 