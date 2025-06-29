from typing import List, Tuple, Dict
import random
import os
import pickle
from agents.td_learning import TDAgent, LookAheadAgent, RandomAgent
from environment import TicTacToeEnv
import random

def evaluate_agent_on_cases(agent: TDAgent, test_cases: List[Tuple[List[str], int]]) -> float:
    """
    Evaluate agent's performance on predefined test cases.
    Each test case represents a critical game state where there's a clear best move.
    Uses evaluation_mode to disable exploration for accurate assessment.
    
    Args:
        agent: The agent to evaluate
        test_cases: List of (board_state, expected_action) tuples
    
    Returns:
        float: Accuracy (proportion of correct decisions)
    """
    correct = 0
    for board, expected_action in test_cases:
        # Use evaluation_mode=True to disable exploration during testing
        if hasattr(agent, 'choose_action') and 'evaluation_mode' in agent.choose_action.__code__.co_varnames:
            action = agent.choose_action(board, evaluation_mode=True)
        else:
            # Fallback for agents that don't support evaluation_mode
            action = agent.choose_action(board)
        
        # Handle both single expected actions and lists of acceptable actions
        if isinstance(expected_action, list):
            if action in expected_action:
                correct += 1
        else:
            if action == expected_action:
                correct += 1
                
    accuracy = correct / len(test_cases) if test_cases else 0
    return accuracy

# Board indices reference:
# 0 | 1 | 2
# ---------
# 3 | 4 | 5
# ---------
# 6 | 7 | 8

# Test cases for X player (X to move, so board has equal X's and O's)
test_cases_X = [
    # Horizontal Wins
    (   # Complete first row: [X, X, -]
        ['X', 'X', '-',
         'O', 'O', '-',
         '-', '-', '-'], 2),  # X should play at 2 to win
         
    (   # Complete middle row: [X, -, X]
        ['O', '-', 'O',
         'X', '-', 'X',
         '-', '-', '-'], 4),  # X should play at 4 to win
         
    (   # Complete bottom row: [-, X, X]
        ['O', 'O', '-',
         '-', '-', '-',
         'X', 'X', '-'], 8),  # X should play at 8 to win

    # Vertical Wins    
    (   # Complete first column: [X, -, X]
        ['X', 'O', '-',
         '-', 'O', '-',
         'X', '-', '-'], 3),  # X should play at 3 to win
         
    (   # Complete diagonal: [X, -, -, -, -, -]
        ['X', 'O', '-',
         '-', 'X', 'O',
         '-', '-', '-'], 8),  # X should play at 8 to win
]

# Test cases for O player (O to move, so board has one more X than O)
test_cases_O = [
    # Horizontal Wins
    (   # Complete middle row: [-, O, O]
        ['X', '-', 'X',
         'O', 'O', '-',
         '-', '-', '-'], 5),  # O should play at 5 to win
         
    (   # Complete first row: [O, -, O]
        ['O', '-', 'O',
         '-', '-', '-',
         'X', 'X', '-'], 1),  # O should play at 1 to win
         
    (   # Complete bottom row: [O, O, -]
        ['X', '-', '-',
         'X', '-', '-',
         'O', 'O', '-'], 8),  # O should play at 8 to win
         
    # Vertical Wins
    (   # Complete first column: [O, -, O]
        ['O', '-', 'X',
         '-', 'X', '-',
         'O', '-', '-'], 3),  # O should play at 3 to win
         
    (   # Complete diagonal: [O, -, -, -, O, -]
        ['O', '-', 'X',
         '-', 'O', '-',
         'X', '-', '-'], 8),  # O should play at 8 to win
]

# Strategic test cases - evaluate positional understanding
strategic_test_cases_X = [
    # Opening: Center control (X moves first, empty board)
    (   # Empty board - should prefer center
        ['-', '-', '-',
         '-', '-', '-',
         '-', '-', '-'], 4),  # X should play center
         
    # Opening: Corner response to center
    (   # O took center, X should take corner
        ['-', '-', '-',
         '-', 'O', '-',
         '-', '-', '-'], [0, 2, 6, 8]),  # Any corner is good
         
    # Fork setup: Create multiple winning threats
    (   # Set up fork - X in corners can create double threat
        ['X', '-', '-',
         '-', 'O', '-',
         '-', '-', 'X'], 1),  # X at 1 creates fork threat (top row + diagonal)
         
    # Fork setup: Alternative fork
    (   # Another fork setup
        ['X', '-', '-',
         '-', 'O', '-',
         '-', '-', 'X'], 7),  # X at 7 creates fork threat (bottom row + diagonal)
         
    # Prevent opponent fork
    (   # O is setting up fork, X must prevent it
        ['-', 'O', '-',
         '-', 'X', '-',
         'O', '-', '-'], 2),  # X should block at 2 to prevent O fork
]

strategic_test_cases_O = [
    # Respond to corner opening
    (   # X took corner, O should take center
        ['X', '-', '-',
         '-', '-', '-',
         '-', '-', '-'], 4),  # O should take center
         
    # Fork setup for O
    (   # O can set up fork
        ['-', 'X', '-',
         '-', 'O', '-',
         '-', 'X', 'O'], 0),  # O at 0 creates fork (top row + left column)
         
    # Prevent X fork
    (   # X is threatening fork, O must block
        ['X', '-', 'X',
         '-', 'O', '-',
         '-', '-', '-'], 1),  # O should block at 1 to prevent X fork
         
    # Advanced positioning
    (   # Force X into bad position
        ['-', '-', '-',
         'X', 'O', '-',
         '-', '-', '-'], 0),  # O should take corner to maintain advantage
]

# Multi-move tactical sequences (2-move combinations)
multi_move_test_cases_X = [
    # Force win in 2 moves
    (   # X can force guaranteed win
        ['X', '-', '-',
         '-', 'X', '-',
         '-', '-', 'O'], 6),  # X at 6 forces win (O must block, then X wins at 2 or 8)
         
    # Setup forced sequence
    (   # Create unstoppable threat
        ['-', 'X', '-',
         'O', 'X', '-',
         '-', '-', '-'], 7),  # X at 7 creates multiple threats
]

multi_move_test_cases_O = [
    # Force win in 2 moves
    (   # O can force guaranteed win
        ['-', 'X', '-',
         '-', 'O', '-',
         'O', 'X', '-'], 0),  # O at 0 forces win sequence
         
    # Counter X's strategy
    (   # Prevent X's forced win
        ['X', '-', '-',
         '-', 'X', 'O',
         '-', '-', '-'], 2),  # O must block at 2 to prevent forced loss
]

def evaluate_both_agents(agent1: TDAgent, agent2: TDAgent, competitor: str) -> Tuple[float, float]:
    """
    Evaluate both agents in both X and O roles.
    Each agent is tested with appropriate test cases for each role.
    
    Args:
        agent1: First agent to evaluate
        agent2: Second agent to evaluate
        competitor: The type of competitor ('random', 'selfplay', or 'independent')
    
    Returns:
        tuple: (agent1 accuracy across both roles, agent2 accuracy across both roles)
    """
    # Save original roles to restore later
    original_role1 = agent1.role
    original_role2 = agent2.role
    
    # Test agent1 in both roles
    agent1.set_role('X')
    agent1_as_X = evaluate_agent_on_cases(agent1, test_cases_X)
    agent1.set_role('O')
    agent1_as_O = evaluate_agent_on_cases(agent1, test_cases_O)
    agent1_accuracy = (agent1_as_X + agent1_as_O) / 2
    
    # Test agent2 in both roles
    agent2.set_role('X')
    agent2_as_X = evaluate_agent_on_cases(agent2, test_cases_X)
    agent2.set_role('O')
    agent2_as_O = evaluate_agent_on_cases(agent2, test_cases_O)
    agent2_accuracy = (agent2_as_X + agent2_as_O) / 2
    
    # Restore original roles
    agent1.set_role(original_role1)
    agent2.set_role(original_role2)
    
    return agent1_accuracy, agent2_accuracy

def evaluate_enhanced(agent1: TDAgent, agent2: TDAgent, competitor: str) -> dict:
    """
    Enhanced evaluation that provides detailed breakdown of different skill types.
    
    Args:
        agent1: First agent to evaluate
        agent2: Second agent to evaluate
        competitor: The type of competitor ('random', 'selfplay', or 'independent')
    
    Returns:
        dict: Detailed evaluation metrics for both agents
    """
    # Save original roles
    original_role1 = agent1.role
    original_role2 = agent2.role
    
    def evaluate_single_agent(agent):
        """Evaluate a single agent across all test categories."""
        # Basic tactical skills (current test cases)
        agent.set_role('X')
        tactical_X = evaluate_agent_on_cases(agent, test_cases_X)
        agent.set_role('O')
        tactical_O = evaluate_agent_on_cases(agent, test_cases_O)
        tactical_avg = (tactical_X + tactical_O) / 2
        
        # Strategic understanding
        agent.set_role('X')
        strategic_X = evaluate_agent_on_cases(agent, strategic_test_cases_X)
        agent.set_role('O')
        strategic_O = evaluate_agent_on_cases(agent, strategic_test_cases_O)
        strategic_avg = (strategic_X + strategic_O) / 2
        
        # Multi-move sequences
        agent.set_role('X')
        multimove_X = evaluate_agent_on_cases(agent, multi_move_test_cases_X)
        agent.set_role('O')
        multimove_O = evaluate_agent_on_cases(agent, multi_move_test_cases_O)
        multimove_avg = (multimove_X + multimove_O) / 2
        
        return {
            'tactical': tactical_avg,
            'strategic': strategic_avg,
            'multi_move': multimove_avg,
            'overall': (tactical_avg + strategic_avg + multimove_avg) / 3,
            'breakdown': {
                'tactical_X': tactical_X,
                'tactical_O': tactical_O,
                'strategic_X': strategic_X,
                'strategic_O': strategic_O,
                'multimove_X': multimove_X,
                'multimove_O': multimove_O
            }
        }
    
    # Evaluate agents
    if competitor == 'selfplay':
        # Same agent in self-play
        agent_results = evaluate_single_agent(agent1)
        results = {
            'agent1': agent_results,
            'agent2': agent_results,  # Same results since same agent
            'mode': 'selfplay'
        }
    else:
        # Different agents
        results = {
            'agent1': evaluate_single_agent(agent1),
            'agent2': evaluate_single_agent(agent2),
            'mode': competitor
        }
    
    # Restore original roles
    agent1.set_role(original_role1)
    agent2.set_role(original_role2)
    
    return results

def play_single_game(agent1, agent2) -> int:
    """
    Play a single game between two agents.
    Returns: 1 if agent1 wins, -1 if agent2 wins, 0 if draw
    """
    env = TicTacToeEnv()
    
    # Randomly assign roles
    agent1_is_X = random.choice([True, False])
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
    second_agent.reset_episode()
    
    current_agent = first_agent
    done = False
    
    while not done:
        action = current_agent.choose_action(state, evaluation_mode=True)
        next_state, reward, done = env.step(action)
        
        if done:
            if reward == 1:  # Someone won
                if current_agent == agent1:
                    return 1  # agent1 won
                else:
                    return -1  # agent2 won
            else:  # Draw
                return 0
        
        state = next_state
        current_agent = second_agent if current_agent == first_agent else first_agent
    
    return 0  # Should not reach here

def quick_benchmark_test(agent, games=10) -> float:
    """
    Quick test of agent vs random opponent.
    Returns win rate (0.0 to 1.0)
    """
    wins = 0
    random_opponent = RandomAgent()
    
    for _ in range(games):
        result = play_single_game(agent, random_opponent)
        if result == 1:  # Agent won
            wins += 1
        # Note: we don't count draws as wins
    
    return wins / games

def evaluate_agent_strength(agent, games_per_opponent=50) -> Dict:
    """Simple strength evaluation - higher win rates = stronger agent"""
    
    opponents = [
        ('Random', RandomAgent()),
        ('Fresh TD', TDAgent()),  # Untrained agent
        ('Fresh LookAhead', LookAheadAgent())  # Untrained lookahead
    ]
    
    strength_score = 0
    results = {}
    
    print("Testing agent strength...")
    for name, opponent in opponents:
        print(f"  vs {name}...", end=" ")
        wins = 0
        draws = 0
        
        for _ in range(games_per_opponent):
            result = play_single_game(agent, opponent)
            if result == 1:  # Agent won
                wins += 1
            elif result == 0:  # Draw
                draws += 1
        
        win_rate = wins / games_per_opponent
        draw_rate = draws / games_per_opponent
        results[name] = {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': 1 - win_rate - draw_rate
        }
        strength_score += win_rate
        print(f"Win: {win_rate:.1%}, Draw: {draw_rate:.1%}")
    
    return {
        'strength_score': strength_score / len(opponents),  # Average win rate
        'detailed': results
    }

def save_agent_checkpoint(agent, episode, checkpoint_dir='checkpoints'):
    """Save agent at specific training episode"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'value_function': agent.value_function.copy(),
        'epsilon': agent.epsilon,
        'episode': episode,
        'agent_type': type(agent).__name__,
        'learning_rate': agent.learning_rate,
        'discount_factor': agent.discount_factor
    }
    
    filename = f"{checkpoint_dir}/agent_episode_{episode}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Saved checkpoint: {filename}")

def load_agent_checkpoint(filename, agent_class=LookAheadAgent):
    """Load agent from checkpoint file"""
    with open(filename, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Create agent with same parameters
    agent = agent_class(
        learning_rate=checkpoint_data['learning_rate'],
        discount_factor=checkpoint_data['discount_factor'],
        epsilon=checkpoint_data['epsilon']
    )
    
    # Restore learned state
    agent.value_function = checkpoint_data['value_function']
    agent.epsilon = checkpoint_data['epsilon']
    
    return agent, checkpoint_data['episode']

def test_generational_improvement(checkpoint_dir='checkpoints', games_per_matchup=20, verbose=True):
    """Load agents from different training stages and see who's stronger"""
    
    # Find all checkpoint files
    checkpoint_files = []
    if os.path.exists(checkpoint_dir):
        for filename in sorted(os.listdir(checkpoint_dir)):
            if filename.endswith('.pkl'):
                checkpoint_files.append(os.path.join(checkpoint_dir, filename))
    
    if len(checkpoint_files) < 2:
        print(f"Need at least 2 checkpoints in {checkpoint_dir}/ to test generational improvement")
        print("Save checkpoints during training using save_agent_checkpoint()")
        return
    
    # Load all agents
    agents = []
    for checkpoint_file in checkpoint_files:
        try:
            agent, episode = load_agent_checkpoint(checkpoint_file)
            agents.append((f"Episode {episode}", agent, episode))
            if verbose:
                print(f"Loaded: {checkpoint_file} (Episode {episode})")
        except Exception as e:
            if verbose:
                print(f"Failed to load {checkpoint_file}: {e}")
    
    if len(agents) < 2:
        if verbose:
            print("Failed to load enough valid checkpoints")
        return
    
    if verbose:
        print(f"\n{'='*60}")
        print("GENERATIONAL TOURNAMENT")
        print(f"{'='*60}")
    
    # Round-robin tournament
    results_matrix = {}
    
    for i, (name1, agent1, ep1) in enumerate(agents):
        results_matrix[name1] = {}
        
        for j, (name2, agent2, ep2) in enumerate(agents):
            if i == j:
                results_matrix[name1][name2] = "---"
                continue
            
            if verbose:
                print(f"\n{name1} vs {name2}:", end=" ")
            
            wins = 0
            draws = 0
            for _ in range(games_per_matchup):
                result = play_single_game(agent1, agent2)
                if result == 1:
                    wins += 1
                elif result == 0:
                    draws += 1
            
            win_rate = wins / games_per_matchup
            draw_rate = draws / games_per_matchup
            
            results_matrix[name1][name2] = f"{win_rate:.1%}"
            if verbose:
                print(f"{win_rate:.1%} wins, {draw_rate:.1%} draws")
    
    if verbose:
        # Print results matrix
        print(f"\n{'='*60}")
        print("RESULTS MATRIX (Row vs Column)")
        print(f"{'='*60}")
        
        # Header
        agent_names = [name for name, _, _ in agents]
        print(f"{'':15}", end="")
        for name in agent_names:
            print(f"{name:>12}", end="")
        print()
        
        # Rows
        for name1 in agent_names:
            print(f"{name1:15}", end="")
            for name2 in agent_names:
                print(f"{results_matrix[name1][name2]:>12}", end="")
            print()
        
        print(f"\n{'='*60}")
        print("OVERALL STRENGTH RANKING")
        print(f"{'='*60}")
    
    # Calculate overall strength (average win rate against all others)
    strength_scores = []
    for name1, agent1, ep1 in agents:
        total_wins = 0
        total_games = 0
        
        for name2, agent2, ep2 in agents:
            if name1 != name2:
                wins = 0
                for _ in range(games_per_matchup):
                    result = play_single_game(agent1, agent2)
                    if result == 1:
                        wins += 1
                total_wins += wins
                total_games += games_per_matchup
        
        overall_strength = total_wins / total_games if total_games > 0 else 0
        strength_scores.append((overall_strength, name1, ep1))
    
    # Sort by strength
    strength_scores.sort(reverse=True)
    
    if verbose:
        for rank, (strength, name, episode) in enumerate(strength_scores, 1):
            print(f"{rank}. {name}: {strength:.1%} overall win rate")
    
    return results_matrix, strength_scores