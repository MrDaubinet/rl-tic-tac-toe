#!/usr/bin/env python3
"""
Model Matrix Benchmark: Latest checkpoint from each model vs every other model.
Creates a comprehensive comparison matrix between different training approaches.
"""

import os
import json
import pickle
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from agents.td_learning import TDAgent, LookAheadAgent, QLearning, RandomAgent
from evaluation import play_single_game

class DualRoleAgent:
    """Agent that can play both X and O roles with different value functions"""
    
    def __init__(self, agent_class, x_value_function, o_value_function, **kwargs):
        self.agent_class = agent_class
        self.x_value_function = x_value_function
        self.o_value_function = o_value_function
        self.kwargs = kwargs
        self._current_agent = None
        self._current_role = None
    
    def _get_agent_for_role(self, role):
        """Get the appropriate agent instance for the given role"""
        if self._current_role != role:
            self._current_agent = self.agent_class(**self.kwargs)
            self._current_agent.epsilon = 0.0  # Evaluation mode
            self._current_agent.set_role(role)  # Set the role on the agent
            if role == 'X':
                self._current_agent.value_function = self.x_value_function
            else:
                self._current_agent.value_function = self.o_value_function
            self._current_role = role
        return self._current_agent
    
    def choose_action(self, state, role='X'):
        """Choose action for the given role"""
        agent = self._get_agent_for_role(role)
        return agent.choose_action(state)
    
    def get_state_value(self, state, role='X'):
        """Get state value for the given role"""
        agent = self._get_agent_for_role(role)
        return agent.get_state_value(state)

class RandomDualRoleAgent:
    """Random agent that can play both X and O roles"""
    
    def __init__(self):
        self.agent = RandomAgent()
    
    def choose_action(self, state, role='X'):
        """Choose random action for any role"""
        self.agent.set_role(role)
        return self.agent.choose_action(state)

class ModelMatrixBenchmark:
    """Benchmark suite that compares latest checkpoints from different model types"""
    
    def __init__(self, weights_dir: str = './weights'):
        self.weights_dir = weights_dir
        
    def load_json_weights(self, x_filepath: str, o_filepath: str, agent_type: str) -> DualRoleAgent:
        """Load agent from JSON weights files for both X and O"""
        with open(x_filepath, 'r') as f:
            x_weights_data = json.load(f)
        
        with open(o_filepath, 'r') as f:
            o_weights_data = json.load(f)
        
        # Determine agent class based on type
        if agent_type == 'LookAheadAgent':
            agent_class = LookAheadAgent
        elif agent_type == 'QLearning':
            agent_class = QLearning
        elif agent_type == 'TDAgent':
            agent_class = TDAgent
        elif agent_type == 'SelfPlayAgent':
            agent_class = TDAgent  # Using TDAgent as fallback
        else:
            agent_class = TDAgent  # Default fallback
            
        # Create dual-role agent
        dual_agent = DualRoleAgent(
            agent_class=agent_class,
            x_value_function=x_weights_data,
            o_value_function=o_weights_data
        )
        
        return dual_agent
    
    def find_model_types(self) -> Dict[str, Dict[str, str]]:
        """Find all model types and their weight files"""
        models = {}
        
        if not os.path.exists(self.weights_dir):
            print(f"Weights directory {self.weights_dir} not found!")
            return models
        
        # Check for paired X/O files in root directory
        json_files = [f for f in os.listdir(self.weights_dir) if f.endswith('.json')]
        
        # Group by model type (everything before _X or _O)
        model_pairs = {}
        for json_file in json_files:
            if '_X.json' in json_file:
                model_name = json_file.replace('_X.json', '')
                if model_name not in model_pairs:
                    model_pairs[model_name] = {}
                model_pairs[model_name]['X'] = os.path.join(self.weights_dir, json_file)
            elif '_O.json' in json_file:
                model_name = json_file.replace('_O.json', '')
                if model_name not in model_pairs:
                    model_pairs[model_name] = {}
                model_pairs[model_name]['O'] = os.path.join(self.weights_dir, json_file)
        
        # Only include models that have both X and O files
        for model_name, files in model_pairs.items():
            if 'X' in files and 'O' in files:
                models[model_name] = files
        
        # Check subdirectories for different training runs
        for subdir in os.listdir(self.weights_dir):
            subdir_path = os.path.join(self.weights_dir, subdir)
            if os.path.isdir(subdir_path):
                # Find JSON files in this subdirectory
                json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                
                # Group by model type within subdirectory
                subdir_pairs = {}
                for json_file in json_files:
                    if '_X.json' in json_file:
                        model_name = json_file.replace('_X.json', '')
                        if model_name not in subdir_pairs:
                            subdir_pairs[model_name] = {}
                        subdir_pairs[model_name]['X'] = os.path.join(subdir_path, json_file)
                    elif '_O.json' in json_file:
                        model_name = json_file.replace('_O.json', '')
                        if model_name not in subdir_pairs:
                            subdir_pairs[model_name] = {}
                        subdir_pairs[model_name]['O'] = os.path.join(subdir_path, json_file)
                
                # Add complete pairs with subdirectory prefix
                for model_name, files in subdir_pairs.items():
                    if 'X' in files and 'O' in files:
                        full_model_name = f"{subdir}_{model_name}"
                        models[full_model_name] = files
        
        return models
    
    def load_model(self, model_name: str, model_files: Dict[str, str]) -> DualRoleAgent:
        """Load a model and return a DualRoleAgent"""
        try:
            # Extract agent type from model name
            if 'QLearning' in model_name:
                agent_type = 'QLearning'
            elif 'LookAhead' in model_name:
                agent_type = 'LookAheadAgent'
            elif 'TD' in model_name:
                agent_type = 'TDAgent'
            elif 'MonteCarlo' in model_name or 'mc_' in model_name:
                agent_type = 'MonteCarloAgent'
            elif 'SelfPlay' in model_name or 'selfplay' in model_name:
                agent_type = 'SelfPlayAgent'
            else:
                agent_type = 'TDAgent'  # Default
            
            agent = self.load_json_weights(
                model_files['X'], 
                model_files['O'], 
                agent_type
            )
            
            return agent
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            return None
    
    def play_game_with_roles(self, agent1, agent2) -> int:
        """Play a game where agent1 is X and agent2 is O"""
        from environment import TicTacToeEnv
        
        env = TicTacToeEnv()
        state = env.reset()
        done = False
        
        while not done:
            if env.current_player == 'X':
                # Agent1 plays as X
                action = agent1.choose_action(state, role='X')
            else:
                # Agent2 plays as O
                action = agent2.choose_action(state, role='O')
            
            # Validate action
            if action < 0 or action >= 9 or state[action] != '-':
                # Invalid move - current player loses
                if env.current_player == 'X':
                    return -1  # Agent2 (O) wins
                else:
                    return 1  # Agent1 (X) wins
            
            state, reward, done = env.step(action)
            
            if done:
                if reward == 1.0:
                    # Current player won
                    if env.current_player == 'X':
                        return 1  # Agent1 (X) won
                    else:
                        return -1  # Agent2 (O) won
                elif reward == 0.5:
                    return 0  # Draw
                else:
                    # Invalid move - opponent wins
                    if env.current_player == 'X':
                        return -1  # Agent2 (O) wins
                    else:
                        return 1  # Agent1 (X) wins
        
        return 0  # Should not reach here
    
    def run_matrix_benchmark(self, games_per_matchup: int = 100) -> Tuple[Dict, Dict, List]:
        """Run comprehensive matrix benchmark between all model types"""
        print("🚀 MODEL MATRIX BENCHMARK")
        print("=" * 80)
        
        # Find all model types
        models = self.find_model_types()
        if not models:
            print("No complete model pairs found!")
            return {}, {}, []
        
        print(f"Found {len(models)} trained model types:")
        for model_name in models.keys():
            print(f"  • {model_name}")
        
        # Load all agents
        all_agents = []
        
        # Add trained models
        for model_name, model_files in models.items():
            agent = self.load_model(model_name, model_files)
            if agent is not None:
                all_agents.append((model_name, agent))
        
        # Add random agent
        random_agent = RandomDualRoleAgent()
        all_agents.append(("Random", random_agent))
        
        if len(all_agents) < 2:
            print("Need at least 2 agents for matrix benchmark")
            return {}, {}, []
        
        print(f"\nLoaded {len(all_agents)} agents total:")
        for agent_name, _ in all_agents:
            print(f"  • {agent_name}")
        print()
        
        # Create results matrix
        results_matrix = {}
        win_counts = {}
        
        print("🎯 RUNNING MATRIX MATCHES")
        print("=" * 80)
        
        for i, (name1, agent1) in enumerate(all_agents):
            results_matrix[name1] = {}
            win_counts[name1] = {'wins': 0, 'draws': 0, 'losses': 0, 'total_games': 0}
            
            for j, (name2, agent2) in enumerate(all_agents):
                if i == j:
                    # Self-play: agent plays against itself
                    print(f"  {name1:25} vs {name2:25} ... ", end="")
                    
                    wins = draws = losses = 0
                    
                    # For self-play, we still alternate roles but it's the same agent
                    for game in range(games_per_matchup):
                        if game % 2 == 0:
                            # Agent1 plays X, Agent2 (same) plays O
                            result = self.play_game_with_roles(agent1, agent2)
                        else:
                            # Agent2 (same) plays X, Agent1 plays O (flip result)
                            result = -self.play_game_with_roles(agent2, agent1)
                        
                        if result == 1:
                            wins += 1
                        elif result == 0:
                            draws += 1
                        else:
                            losses += 1
                    
                    win_rate = wins / games_per_matchup
                    draw_rate = draws / games_per_matchup
                    loss_rate = losses / games_per_matchup
                    
                    results_matrix[name1][name2] = {
                        'win_rate': win_rate,
                        'draw_rate': draw_rate,
                        'loss_rate': loss_rate,
                        'wins': wins,
                        'draws': draws,
                        'losses': losses
                    }
                    
                    # Update win counts (self-play contributes to overall stats)
                    win_counts[name1]['wins'] += wins
                    win_counts[name1]['draws'] += draws
                    win_counts[name1]['losses'] += losses
                    win_counts[name1]['total_games'] += games_per_matchup
                    
                    print(f"W:{win_rate:5.1%} D:{draw_rate:5.1%} L:{loss_rate:5.1%}")
                    continue
                
                print(f"  {name1:25} vs {name2:25} ... ", end="")
                
                wins = draws = losses = 0
                
                # Play games with both role assignments
                for game in range(games_per_matchup):
                    if game % 2 == 0:
                        # Agent1 plays X, Agent2 plays O
                        result = self.play_game_with_roles(agent1, agent2)
                    else:
                        # Agent2 plays X, Agent1 plays O (flip result)
                        result = -self.play_game_with_roles(agent2, agent1)
                    
                    if result == 1:
                        wins += 1
                    elif result == 0:
                        draws += 1
                    else:
                        losses += 1
                
                win_rate = wins / games_per_matchup
                draw_rate = draws / games_per_matchup
                loss_rate = losses / games_per_matchup
                
                results_matrix[name1][name2] = {
                    'win_rate': win_rate,
                    'draw_rate': draw_rate,
                    'loss_rate': loss_rate,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses
                }
                
                # Update win counts
                win_counts[name1]['wins'] += wins
                win_counts[name1]['draws'] += draws
                win_counts[name1]['losses'] += losses
                win_counts[name1]['total_games'] += games_per_matchup
                
                print(f"W:{win_rate:5.1%} D:{draw_rate:5.1%} L:{loss_rate:5.1%}")
        
        return results_matrix, win_counts, all_agents
    
    def print_results_matrix(self, results_matrix: Dict, win_counts: Dict, all_agents: List):
        """Print formatted results matrix with improved readability"""
        agent_names = [name for name, _ in all_agents]
        
        print(f"\n" + "="*120)
        print(f"📊 COMPREHENSIVE RESULTS MATRIX")
        print(f"="*120)
        print(f"Each cell shows: Win% | Draw% | Loss%")
        print(f"Row agent vs Column agent")
        print(f"Self-play results shown in [brackets] - measures consistency when agent plays both X and O")
        print(f"="*120)
        
        # Calculate column widths
        name_width = max(len(name) for name in agent_names) + 2
        cell_width = 12
        
        # Print header
        print(f"{'Agent':<{name_width}}", end="")
        for name in agent_names:
            short_name = name[:10] if len(name) > 10 else name
            print(f"{short_name:^{cell_width}}", end="")
        print(f"{'Overall':^{cell_width}}")
        
        # Print separator
        print("-" * (name_width + cell_width * (len(agent_names) + 1)))
        
        # Print each row
        for name1 in agent_names:
            print(f"{name1:<{name_width}}", end="")
            
            for name2 in agent_names:
                result = results_matrix[name1][name2]
                w = int(result['win_rate'] * 100)
                d = int(result['draw_rate'] * 100)
                l = int(result['loss_rate'] * 100)
                if name1 == name2:
                    # Self-play: highlight with different format
                    cell_text = f"[{w:2d}|{d:2d}|{l:2d}]"
                else:
                    cell_text = f"{w:2d}|{d:2d}|{l:2d}"
                print(f"{cell_text:^{cell_width}}", end="")
            
            # Calculate overall win rate
            total_wins = win_counts[name1]['wins']
            total_games = win_counts[name1]['total_games']
            overall_rate = int((total_wins / total_games * 100)) if total_games > 0 else 0
            print(f"{overall_rate:^{cell_width}}")
        
        print()
        
        # Print win rate matrix (simplified)
        print(f"\n" + "="*80)
        print(f"🏆 WIN RATE MATRIX (Percentage)")
        print(f"="*80)
        
        # Print header
        print(f"{'Agent':<{name_width}}", end="")
        for name in agent_names:
            short_name = name[:8] if len(name) > 8 else name
            print(f"{short_name:>8}", end="")
        print(f"{'Overall':>8}")
        
        # Print separator
        print("-" * (name_width + 8 * (len(agent_names) + 1)))
        
        # Print each row
        for name1 in agent_names:
            print(f"{name1:<{name_width}}", end="")
            
            for name2 in agent_names:
                result = results_matrix[name1][name2]
                win_pct = int(result['win_rate'] * 100)
                if name1 == name2:
                    # Self-play: show in brackets
                    print(f"[{win_pct:>2}%]", end="")
                else:
                    print(f"{win_pct:>7}%", end="")
            
            # Calculate overall win rate
            total_wins = win_counts[name1]['wins']
            total_games = win_counts[name1]['total_games']
            overall_rate = int((total_wins / total_games * 100)) if total_games > 0 else 0
            print(f"{overall_rate:>7}%")
        
        # Print ranking
        print(f"\n" + "="*60)
        print(f"🥇 AGENT RANKINGS")
        print(f"="*60)
        
        # Sort agents by overall performance
        sorted_agents = sorted(win_counts.items(), 
                             key=lambda x: x[1]['wins'] / max(x[1]['total_games'], 1), 
                             reverse=True)
        
        print(f"{'Rank':>4} {'Agent':<25} {'Win Rate':>10} {'Draw Rate':>10} {'Total Games':>12}")
        print("-" * 60)
        
        for rank, (name, stats) in enumerate(sorted_agents, 1):
            if stats['total_games'] > 0:
                win_rate = stats['wins'] / stats['total_games']
                draw_rate = stats['draws'] / stats['total_games']
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"{medal:>4} {name:<25} {win_rate:>9.1%} {draw_rate:>9.1%} {stats['total_games']:>12}")
        
        print()
        
        # Print insights
        print(f"\n" + "="*60)
        print(f"💡 KEY INSIGHTS")
        print(f"="*60)
        
        # Find best and worst performers
        best_agent = sorted_agents[0]
        worst_agent = sorted_agents[-1]
        
        print(f"🏆 Best Performer: {best_agent[0]}")
        best_win_rate = best_agent[1]['wins'] / best_agent[1]['total_games']
        best_draw_rate = best_agent[1]['draws'] / best_agent[1]['total_games']
        print(f"   Win Rate: {best_win_rate:.1%}, Draw Rate: {best_draw_rate:.1%}")
        
        print(f"\n📉 Needs Improvement: {worst_agent[0]}")
        worst_win_rate = worst_agent[1]['wins'] / worst_agent[1]['total_games']
        worst_draw_rate = worst_agent[1]['draws'] / worst_agent[1]['total_games']
        print(f"   Win Rate: {worst_win_rate:.1%}, Draw Rate: {worst_draw_rate:.1%}")
        
        # Find highest draw rate
        highest_draw = max(sorted_agents, key=lambda x: x[1]['draws'] / max(x[1]['total_games'], 1))
        highest_draw_rate = highest_draw[1]['draws'] / highest_draw[1]['total_games']
        print(f"\n🤝 Most Draws: {highest_draw[0]} ({highest_draw_rate:.1%})")
        
        # Self-play analysis
        print(f"\n🔄 SELF-PLAY ANALYSIS:")
        for name in agent_names:
            self_result = results_matrix[name][name]
            self_draw_rate = self_result['draw_rate']
            self_win_rate = self_result['win_rate']
            consistency = "High" if self_draw_rate > 0.6 else "Medium" if self_draw_rate > 0.3 else "Low"
            print(f"   {name}: {self_draw_rate:.1%} draws, {self_win_rate:.1%} wins - {consistency} consistency")
        
        print()

def main():
    parser = argparse.ArgumentParser(description='Run model matrix benchmark')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games per matchup (default: 100)')
    parser.add_argument('--weights-dir', type=str, default='./weights',
                       help='Directory containing model weights (default: ./weights)')
    
    args = parser.parse_args()
    
    benchmark = ModelMatrixBenchmark(args.weights_dir)
    results_matrix, win_counts, all_agents = benchmark.run_matrix_benchmark(args.games)
    
    if results_matrix:
        benchmark.print_results_matrix(results_matrix, win_counts, all_agents)
    
    print("\n✅ Matrix benchmark complete!")

if __name__ == "__main__":
    main() 