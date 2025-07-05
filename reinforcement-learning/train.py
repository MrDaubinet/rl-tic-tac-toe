import numpy as np
from typing import List, Tuple
import os
import argparse
from environment import TicTacToeEnv
from agents.td_agent import TDAgent
from agents.minmax_agent import MinMaxTDAgent
from agents.qlearning_agent import QLearning
from agents.random_agent import RandomAgent
from visualization import LearningVisualizer
from evaluation import evaluate_both_agents, save_agent_checkpoint, load_agent_checkpoint

def get_model(model: str, lookahead_depth: int = 2):
    if model == 'td':
        return lambda **kwargs: TDAgent(**kwargs)
    elif model == 'minmaxtd':
        return lambda **kwargs: MinMaxTDAgent(lookahead_depth=lookahead_depth, **kwargs)
    elif model == 'qlearning':
        return lambda **kwargs: QLearning(lookahead_depth=lookahead_depth, **kwargs)
    elif model == 'random':
        return lambda **kwargs: RandomAgent(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model}")

def train_agents(episodes: int,
                save_dir: str,
                visualize_every: int,
                model: str,
                weights_dir: str = 'weights',
                decay_rate: float = 0.9999,
                min_epsilon: float = 0.01,
                competitor: str = 'random',
                debug: bool = False,
                lookahead_depth: int = 2,
                learning_rate: float = None,
                discount_factor: float = None,
                epsilon: float = None
                ) -> None:
    """
    Train agent(s) for Tic-Tac-Toe. Supports four modes:
    1. Self-play: Same agent plays against itself
    2. Random opponent: Agent plays against random moves
    3. Two separate agents: Different agents play against each other
    4. TD agent competitor: Agent plays against a TD learning agent
    
    For LookAhead agents, lookahead_depth controls minimax search depth:
    - Depth 1: Only considers immediate moves
    - Depth 2: Considers opponent's best response (default)
    - Depth 3+: Deeper search (slower but potentially stronger)
    
    Agents are assigned X/O roles randomly at the start of each episode.
    X always moves first.
    """
    env = TicTacToeEnv()
    
    agent_params = {
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'epsilon': epsilon
    }
    
    # Initialize agents based on competitor type
    model_factory = get_model(model, lookahead_depth)
    
    if competitor == 'selfplay':
        agent1 = model_factory(**agent_params)
        agent2 = agent1  # Use same agent instance
    elif competitor == 'random':
        agent1 = model_factory(**agent_params)
        agent2 = RandomAgent()
    elif competitor == 'independent':
        agent1 = model_factory(**agent_params)
        agent2 = model_factory(**agent_params)
    elif competitor == 'td':
        agent1 = model_factory(**agent_params)
        agent2 = TDAgent(**agent_params)
        print("Using fresh TD agent as competitor")
    elif competitor == 'lookahead':
        agent1 = model_factory(**agent_params)
        agent2 = LookAheadAgent(lookahead_depth=lookahead_depth, **agent_params)
        print("Using fresh LookAhead agent as competitor")
    else:
        raise ValueError(f"Unknown competitor type: {competitor}")
    
    # Debug: Print agent types and configuration
    print(f"Agent1 type: {type(agent1).__name__}")
    print(f"Agent2 type: {type(agent2).__name__}")
    print(f"Competitor mode: {competitor}")
    print(f"Hyperparameters: lr={agent_params['learning_rate']}, γ={agent_params['discount_factor']}, ε={agent_params['epsilon']}")
    if hasattr(agent1, 'lookahead_depth'):
        print(f"Agent1 lookahead depth: {agent1.lookahead_depth}")
    if hasattr(agent2, 'lookahead_depth'):
        print(f"Agent2 lookahead depth: {agent2.lookahead_depth}")
    print("-" * 40)
    
    visualizer = LearningVisualizer()
    os.makedirs(save_dir, exist_ok=True)
    
    # Track wins by agent instead of position
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    total_games = 0
    
    # Track additional metrics for debugging
    wins_as_X = 0  # Track X wins regardless of agent
    wins_as_O = 0  # Track O wins regardless of agent

    for episode in range(1, episodes + 1):
        # Randomly assign roles (X always moves first)
        agent1_is_X = np.random.random() < 0.5
        if agent1_is_X:
            first_agent, second_agent = agent1, agent2
            agent1.set_role('X')
            agent2.set_role('O')
        else:
            first_agent, second_agent = agent2, agent1
            agent2.set_role('X')
            agent1.set_role('O')
        
        state = env.reset()
        first_agent.reset_episode()
        if competitor != 'selfplay':
            second_agent.reset_episode()
        
        done = False
        reward_first = 0
        reward_second = 0
        
        # X (first_agent) always moves first
        current_agent = first_agent
        while not done:
            # Get action from current agent
            action = current_agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Update agent's history and value function
            if isinstance(current_agent, (TDAgent, MinMaxTDAgent, QLearning)):  # Only update learning agents
                state_key = current_agent.get_state_key(state)
                next_state_key = current_agent.get_state_key(next_state)
                current_agent.add_to_history(state_key, action, reward)
                current_agent.update(state_key, next_state_key, reward)
            
            if done:
                total_games += 1
                if reward == 1:  # Win
                    # Determine who won based on environment's current_player
                    winner_role = env.current_player  # This is who just won
                    
                    # Track which agent won based on their roles
                    if (agent1_is_X and winner_role == 'X') or (not agent1_is_X and winner_role == 'O'):
                        wins_agent1 += 1
                    else:
                        wins_agent2 += 1
                    
                    # Track X vs O wins for debugging
                    if winner_role == 'X':
                        wins_as_X += 1
                        reward_first = 1   # X (first player) won
                        reward_second = -1 # O (second player) lost
                    else:  # winner_role == 'O'
                        wins_as_O += 1
                        reward_first = -1  # X (first player) lost
                        reward_second = 1  # O (second player) won
                        
                elif reward == 0.5:  # Draw
                    draws += 1
                    reward_first = 0.5
                    reward_second = 0.5
                
                # End of episode updates for learning agents
                if isinstance(first_agent, (TDAgent, MinMaxTDAgent, QLearning)):
                    first_agent.end_of_episode_update(reward_first)
                if isinstance(second_agent, (TDAgent, MinMaxTDAgent, QLearning)) and competitor != 'selfplay':
                    second_agent.end_of_episode_update(reward_second)
            
            state = next_state
            current_agent = second_agent if current_agent == first_agent else first_agent
        
        # Decay epsilon for learning agents
        if isinstance(agent1, (TDAgent, MinMaxTDAgent, QLearning)):
            agent1.decay_epsilon(decay_rate, min_epsilon)
        if isinstance(agent2, (TDAgent, MinMaxTDAgent, QLearning)) and competitor != 'selfplay':
            agent2.decay_epsilon(decay_rate, min_epsilon)
        
        # Visualization updates
        if episode % visualize_every == 0:
            win_rate1 = wins_agent1 / total_games
            win_rate2 = wins_agent2 / total_games
            draw_rate = draws / total_games
            
            # Map first/second player rewards to agent1/agent2 rewards
            if agent1_is_X:
                # agent1 is X (first), agent2 is O (second)
                reward_agent1 = reward_first
                reward_agent2 = reward_second
            else:
                # agent2 is X (first), agent1 is O (second)
                reward_agent1 = reward_second
                reward_agent2 = reward_first
            
            visualizer.update_win_rates(episode, win_rate1, win_rate2)
            visualizer.update_rewards_both(episode, reward_agent1, reward_agent2)
            
            # Track draw rate for analysis
            if not hasattr(visualizer, 'draw_rates'):
                visualizer.draw_rates = []
            visualizer.draw_rates.append(draw_rate)
            
            # Evaluate agents only on winning test cases for their role
            accuracy1, accuracy2 = evaluate_both_agents(agent1, agent2, competitor)
            visualizer.update_accuracy(episode, accuracy1, accuracy2)
            
            # Save checkpoints every 2nd visualization interval
            if episode % (visualize_every * 2) == 0:
                save_agent_checkpoint(agent1, episode)
            
            print(f"Episode {episode}/{episodes}")
            if competitor == 'selfplay':
                # In self-play, show different metrics since it's the same agent
                decisive_rate = (wins_agent1 + wins_agent2) / total_games
                x_win_rate = wins_as_X / total_games
                o_win_rate = wins_as_O / total_games
                print(f"Agent Learning Progress - Decisive Games: {decisive_rate:.2f} | Draws: {draw_rate:.2f}")
                print(f"X Wins: {x_win_rate:.2f} | O Wins: {o_win_rate:.2f} | Episode Reward: {reward_agent1:.2f}")
                print(f"Test Case Accuracy: {accuracy1:.2f}")
            else:
                print(f"Agent1 Win Rate: {win_rate1:.2f} | Agent2 Win Rate: {win_rate2:.2f} | Draws: {draw_rate:.2f}")
                print(f"Episode Reward - Agent1: {reward_agent1:.2f} | Agent2: {reward_agent2:.2f}")
                print(f"Test Case Accuracy - Agent1: {accuracy1:.2f} | Agent2: {accuracy2:.2f}")
                    
            if debug:
                print(f"Agent1 Epsilon: {agent1.epsilon:.4f}")
                if hasattr(agent1, 'action_value_function'):
                    print(f"Agent1 Q-Function Size: {len(agent1.action_value_function)}")
                else:
                    print(f"Agent1 Value Function Size: {len(agent1.value_function)}")
                x_win_rate = wins_as_X / total_games
                o_win_rate = wins_as_O / total_games
                print(f"Debug - X Win Rate: {x_win_rate:.2f} | O Win Rate: {o_win_rate:.2f}")
            print("-" * 40)
    
    # Save final results
    if isinstance(agent1, (TDAgent, MinMaxTDAgent, QLearning)):
        agent1.save(weights_dir)
        save_agent_checkpoint(agent1, episodes)  # Save final checkpoint
    if isinstance(agent2, (TDAgent, MinMaxTDAgent, QLearning)) and competitor != 'selfplay':
        agent2.save(weights_dir)
    
    # Final statistics
    final_decisive_rate = (wins_agent1 + wins_agent2) / total_games
    final_draw_rate = draws / total_games
    final_x_win_rate = wins_as_X / total_games
    final_o_win_rate = wins_as_O / total_games
    
    print(f"\n{'='*50}")
    print("🎉 Training Complete!")
    print(f"{'='*50}")
    print(f"Final Statistics (over {total_games} games):")
    print(f"  Decisive Games: {final_decisive_rate:.1%}")
    print(f"  Draw Rate: {final_draw_rate:.1%}")
    if competitor == 'selfplay':
        print(f"  X Win Rate: {final_x_win_rate:.1%}")
        print(f"  O Win Rate: {final_o_win_rate:.1%}")
        print(f"  First Player Advantage: {final_x_win_rate - final_o_win_rate:+.1%}")
    else:
        print(f"  Agent1 Win Rate: {wins_agent1/total_games:.1%}")
        print(f"  Agent2 Win Rate: {wins_agent2/total_games:.1%}")
    print(f"\nFiles saved:")
    print(f"  Final agent: {weights_dir}")
    print(f"  Checkpoints: checkpoints/")
    print(f"  Run benchmark.py for comprehensive evaluation")
    
    visualizer.plot_learning_curves(os.path.join(save_dir, 'learning_curves.png'))
    visualizer.plot_accuracy_curve(os.path.join(save_dir, 'accuracy_curve.png'))
    visualizer.save_data(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agents for Tic-Tac-Toe.")
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--visualize_every', type=int, default=1000, help='Visualization interval')
    parser.add_argument('--model', type=str, choices=['td', 'minmaxtd', 'qlearning', 'random'], default='td', help='Agent type')
    parser.add_argument('--save_dir', type=str, default='training_data', help='Directory to save training data')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save agent weights')
    parser.add_argument('--decay_rate', type=float, default=0.9999, help='Epsilon decay rate per episode')
    parser.add_argument('--min_epsilon', type=float, default=0.01, help='Minimum epsilon value')
    parser.add_argument('--competitor', type=str, choices=['random', 'selfplay', 'independent', 'td', 'lookahead'], default='random', 
                      help='Competitor type: random, selfplay, independent, td, or lookahead')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug output (epsilon, value function size, agent2 skills)')
    parser.add_argument('--lookahead_depth', type=int, default=2,
                        help='Lookahead depth for LookAhead agent (1-4 recommended)')
    
    # Hyperparameter arguments
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='Learning rate (default: 0.01, recommend 0.1 for qlearning)')
    parser.add_argument('--discount_factor', type=float, default=0.9,
                        help='Discount factor (default: 0.9, recommend 0.95 for qlearning)')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Initial epsilon for exploration (default: 0.5, recommend 0.9 for qlearning)')
    
    args = parser.parse_args()
    save_subdir = f"{args.save_dir}/{args.model}_{args.competitor}"
    train_agents(
        episodes=args.episodes,
        visualize_every=args.visualize_every,
        model=args.model,
        save_dir=save_subdir,
        weights_dir=args.weights_dir,
        decay_rate=args.decay_rate,
        min_epsilon=args.min_epsilon,
        competitor=args.competitor,
        debug=args.debug,
        lookahead_depth=args.lookahead_depth,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon
    ) 