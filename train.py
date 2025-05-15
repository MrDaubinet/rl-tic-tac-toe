import numpy as np
from typing import List, Tuple
from rl_agents.td_learning import TDAgent, LookAheadAgent, MonteCarloAgent
from rl_agents.visualization import LearningVisualizer
import os
import argparse

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
            return self.board.copy(), -1, True  # Invalid move
        
        self.board[action] = self.current_player
        
        # Check for win
        if self._check_win():
            return self.board.copy(), 1.0, True
        
        # Check for draw
        if '-' not in self.board:
            return self.board.copy(), 0.5, True
        
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

def get_model(model: str):
    if model == 'td':
        return TDAgent
    elif model == 'lookahead':
        return LookAheadAgent
    elif model == 'mc':
        return MonteCarloAgent
    else:
        raise ValueError(f"Unknown model type: {model}")

def train_agent(episodes: int,
                save_dir: str,
                visualize_every: int,
                model: str,
                weights_dir: str = 'weights',
                decay_rate: float = 0.9999,
                min_epsilon: float = 0.01,
                random_agent: bool = False) -> None:
    """
    Train two agents via self-play and visualize their progress.
    """
    AgentClass = get_model(model)
    env = TicTacToeEnv()
    # Use same parameters for both agents
    agent_params = {
        'learning_rate': 0.01,
        'discount_factor': 0.9,
        'epsilon': 0.5
    }
    agent_O = AgentClass(**agent_params, player='O')
    agent_X = AgentClass(**agent_params, player='X')
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
        invalid_moves_O = 0
        invalid_moves_X = 0
        agent_X.reset_episode()
        agent_O.reset_episode()
        # Randomize starting player
        current_player = np.random.choice(['O', 'X'])
        while not done:
            if current_player == 'X':
                action = agent_X.choose_action(state)
                next_state, reward, done = env.step(action)
                state_key = agent_X.get_state_key(state)
                next_state_key = agent_X.get_state_key(next_state)
                agent_X.add_to_history(state_key, action, reward)
                agent_X.update(state_key, next_state_key, reward)

                if done:
                    agent_X.end_of_episode_update(reward)
                    episode_reward_X += reward
                    if reward == 1:
                        wins_X += 1
                    elif reward == 0.5:
                        draws += 1
                    break
                state = next_state
                current_player = 'O'
            else:
                if random_agent:
                    # Random opponent's turn (O)
                    valid_moves = [i for i, mark in enumerate(state) if mark == '-']
                    if valid_moves:
                        o_action = np.random.choice(valid_moves)
                        state, reward, done = env.step(o_action)
                        
                        if done:
                            episode_reward_O += reward
                            if reward == 1:
                                wins_O += 1
                            elif reward == 0.5:
                                draws += 1
                            break
                else:
                    action = agent_O.choose_action(state)
                    next_state, reward, done = env.step(action)
                    state_key = agent_O.get_state_key(state)
                    next_state_key = agent_O.get_state_key(next_state)
                    agent_O.add_to_history(state_key, action, reward)
                    agent_O.update(state_key, next_state_key, reward)
                    
                    if done:
                        agent_O.end_of_episode_update(reward)
                        episode_reward_O += reward
                        if reward == 1:
                            wins_O += 1
                        elif reward == 0.5:
                            draws += 1
                        break

                    state = next_state
                current_player = 'X'
                
        total_games += 1
        # Epsilon decay for both agents
        agent_O.decay_epsilon(decay_rate=decay_rate, min_epsilon=min_epsilon)
        agent_X.decay_epsilon(decay_rate=decay_rate, min_epsilon=min_epsilon)
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
            print(f"Invalid moves: O={invalid_moves_O}, X={invalid_moves_X}")
            print(f"Value function range: [{values.min():.2f}, {values.max():.2f}]")
            print("Saved: learning curves, value heatmap, policy heatmap, value distribution.")
            print("-" * 40)

    # After training, save both agents to the weights directory
    # agent_O.save(directory=weights_dir)
    agent_X.save(directory=weights_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TD or Self-Play agent for Tic-Tac-Toe.")
    parser.add_argument('--episodes', type=int, default=50000, help='Number of training episodes')
    parser.add_argument('--visualize_every', type=int, default=1000, help='Visualization interval')
    parser.add_argument('--model', type=str, choices=['td', 'lookahead'], default='td', help="Agent type: 'td', 'selfplay', or 'mc'")
    parser.add_argument('--save_dir', type=str, default='training_data', help='Directory to save training data')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save agent weights')
    parser.add_argument('--decay_rate', type=float, default=0.9999, help='Epsilon decay rate per episode')
    parser.add_argument('--min_epsilon', type=float, default=0.01, help='Minimum epsilon value')
    parser.add_argument('--random_agent', type=bool, default=False, help='Random agent')
    parser.add_argument('--selfplay', type=bool, default=False, help='Self-play')
    args = parser.parse_args()
    train_agent(
        episodes=args.episodes, 
        visualize_every=args.visualize_every, 
        model=args.model, 
        save_dir=args.save_dir, 
        weights_dir=args.weights_dir, 
        decay_rate=args.decay_rate, 
        min_epsilon=args.min_epsilon,
        random_agent=args.random_agent
    ) 