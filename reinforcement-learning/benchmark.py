#!/usr/bin/env python3
"""
Comprehensive benchmark system for evaluating trained RL agents.
Loads checkpoints and compares performance across different opponents and test scenarios.
"""

import os
import pickle
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from agents.td_learning import TDAgent, LookAheadAgent, RandomAgent
from evaluation import (
    evaluate_enhanced, play_single_game,
    TACTICAL_TEST_CASES_X, TACTICAL_TEST_CASES_O,
    STRATEGIC_TEST_CASES_X, STRATEGIC_TEST_CASES_O,
    MULTI_MOVE_TEST_CASES_X, MULTI_MOVE_TEST_CASES_O
)

class BenchmarkSuite:
    """Comprehensive benchmark suite for RL agents"""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        self.baseline_agents = {
            'Random': RandomAgent(),
            'Fresh TD': TDAgent(),
            'Fresh LookAhead': LookAheadAgent()
        }
        
    def load_checkpoint(self, filename: str) -> Tuple[TDAgent, int]:
        """Load agent from checkpoint file"""
        with open(filename, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Determine agent class from saved type
        if checkpoint_data['agent_type'] == 'LookAheadAgent':
            agent_class = LookAheadAgent
        else:
            agent_class = TDAgent
            
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
    
    def find_checkpoints(self) -> List[Tuple[str, str, int]]:
        """Find all checkpoint files and return (name, filepath, episode)"""
        checkpoints = []
        if not os.path.exists(self.checkpoint_dir):
            print(f"Checkpoint directory {self.checkpoint_dir} not found!")
            return checkpoints
            
        for filename in sorted(os.listdir(self.checkpoint_dir)):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                try:
                    _, episode = self.load_checkpoint(filepath)
                    name = f"Episode {episode}"
                    checkpoints.append((name, filepath, episode))
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
        
        return checkpoints
    
    def evaluate_single_agent(self, agent, agent_name: str, games_per_opponent: int = 100) -> Dict:
        """Evaluate a single agent against all baseline opponents"""
        print(f"\n📊 Evaluating {agent_name}")
        print("-" * 50)
        
        results = {}
        total_strength = 0
        
        # Test against baseline opponents
        for opponent_name, opponent in self.baseline_agents.items():
            print(f"  vs {opponent_name:15}...", end=" ")
            wins = draws = losses = 0
            
            for _ in range(games_per_opponent):
                result = play_single_game(agent, opponent)
                if result == 1:
                    wins += 1
                elif result == 0:
                    draws += 1
                else:
                    losses += 1
            
            win_rate = wins / games_per_opponent
            draw_rate = draws / games_per_opponent
            loss_rate = losses / games_per_opponent
            
            results[opponent_name] = {
                'wins': wins,
                'draws': draws, 
                'losses': losses,
                'win_rate': win_rate,
                'draw_rate': draw_rate,
                'loss_rate': loss_rate
            }
            
            total_strength += win_rate
            print(f"W:{win_rate:5.1%} D:{draw_rate:5.1%} L:{loss_rate:5.1%}")
        
        # Enhanced skill evaluation
        print(f"  Skill Analysis...", end=" ")
        enhanced_results = evaluate_enhanced(agent, RandomAgent(), 'random')
        skills = enhanced_results['agent1']
        
        results['skills'] = skills
        results['overall_strength'] = total_strength / len(self.baseline_agents)
        
        print(f"T:{skills['tactical']:4.1%} S:{skills['strategic']:4.1%} M:{skills['multi_move']:4.1%}")
        
        return results
    
    def head_to_head_tournament(self, checkpoints: List[Tuple[str, str, int]], 
                               games_per_matchup: int = 50) -> Tuple[Dict, List]:
        """Run round-robin tournament between all checkpoints"""
        print(f"\n🏆 HEAD-TO-HEAD TOURNAMENT")
        print(f"{'='*70}")
        
        agents = []
        for name, filepath, episode in checkpoints:
            agent, _ = self.load_checkpoint(filepath)
            agents.append((name, agent, episode))
        
        if len(agents) < 2:
            print("Need at least 2 agents for tournament")
            return {}, []
        
        # Round-robin results matrix
        results_matrix = {}
        
        print(f"Running {len(agents)} x {len(agents)} tournament...")
        for i, (name1, agent1, ep1) in enumerate(agents):
            results_matrix[name1] = {}
            
            for j, (name2, agent2, ep2) in enumerate(agents):
                if i == j:
                    results_matrix[name1][name2] = "---"
                    continue
                
                print(f"  {name1} vs {name2}...", end=" ")
                wins = draws = 0
                
                for _ in range(games_per_matchup):
                    result = play_single_game(agent1, agent2)
                    if result == 1:
                        wins += 1
                    elif result == 0:
                        draws += 1
                
                win_rate = wins / games_per_matchup
                draw_rate = draws / games_per_matchup
                results_matrix[name1][name2] = f"{win_rate:.1%}"
                print(f"{win_rate:.1%} wins")
        
        # Calculate overall tournament strength
        strength_scores = []
        for name1, agent1, ep1 in agents:
            total_wins = total_games = 0
            
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
        
        strength_scores.sort(reverse=True)
        
        return results_matrix, strength_scores
    
    def print_tournament_results(self, results_matrix: Dict, strength_scores: List):
        """Print formatted tournament results"""
        if not results_matrix or not strength_scores:
            return
            
        print(f"\n📋 TOURNAMENT RESULTS MATRIX")
        print(f"{'='*70}")
        
        # Header
        agent_names = list(results_matrix.keys())
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
        
        print(f"\n🏅 TOURNAMENT RANKINGS")
        print(f"{'='*70}")
        for rank, (strength, name, episode) in enumerate(strength_scores, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            print(f"{medal:3} {name:20} {strength:6.1%} overall win rate")
    
    def skill_progression_analysis(self, checkpoints: List[Tuple[str, str, int]]):
        """Analyze how skills develop over training"""
        print(f"\n📈 SKILL PROGRESSION ANALYSIS")
        print(f"{'='*70}")
        
        skill_data = []
        for name, filepath, episode in sorted(checkpoints, key=lambda x: x[2]):
            agent, _ = self.load_checkpoint(filepath)
            enhanced_results = evaluate_enhanced(agent, RandomAgent(), 'random')
            skills = enhanced_results['agent1']
            skill_data.append((episode, skills))
        
        print(f"{'Episode':>8} {'Tactical':>9} {'Strategic':>10} {'Multi-move':>11} {'Overall':>9}")
        print("-" * 60)
        
        for episode, skills in skill_data:
            overall = (skills['tactical'] + skills['strategic'] + skills['multi_move']) / 3
            print(f"{episode:>8} {skills['tactical']:>8.1%} {skills['strategic']:>9.1%} "
                  f"{skills['multi_move']:>10.1%} {overall:>8.1%}")
    
    def run_comprehensive_benchmark(self, games_per_opponent: int = 100, 
                                   games_per_matchup: int = 50):
        """Run the complete benchmark suite"""
        print(f"🚀 COMPREHENSIVE RL AGENT BENCHMARK")
        print(f"{'='*70}")
        
        # Find all checkpoints
        checkpoints = self.find_checkpoints()
        if not checkpoints:
            print("No checkpoints found! Train some agents first.")
            return
        
        print(f"Found {len(checkpoints)} checkpoints:")
        for name, _, episode in checkpoints:
            print(f"  • {name}")
        
        # Individual agent evaluation
        print(f"\n🎯 INDIVIDUAL AGENT EVALUATION")
        print(f"{'='*70}")
        
        all_results = {}
        for name, filepath, episode in checkpoints:
            agent, _ = self.load_checkpoint(filepath)
            results = self.evaluate_single_agent(agent, name, games_per_opponent)
            all_results[name] = results
        
        # Tournament between checkpoints
        if len(checkpoints) > 1:
            results_matrix, strength_scores = self.head_to_head_tournament(
                checkpoints, games_per_matchup)
            self.print_tournament_results(results_matrix, strength_scores)
        
        # Skill progression analysis
        if len(checkpoints) > 1:
            self.skill_progression_analysis(checkpoints)
        
        # Summary and recommendations
        self.print_summary_and_recommendations(all_results, checkpoints)
    
    def print_summary_and_recommendations(self, all_results: Dict, 
                                        checkpoints: List[Tuple[str, str, int]]):
        """Print final summary and training recommendations"""
        print(f"\n💡 SUMMARY & RECOMMENDATIONS")
        print(f"{'='*70}")
        
        if not all_results:
            return
        
        # Find best performing agent
        best_agent = max(all_results.items(), 
                        key=lambda x: x[1]['overall_strength'])
        best_name, best_results = best_agent
        
        print(f"🏆 Best Overall Agent: {best_name}")
        print(f"   Overall Strength: {best_results['overall_strength']:.1%}")
        print(f"   Skills - Tactical: {best_results['skills']['tactical']:.1%}, "
              f"Strategic: {best_results['skills']['strategic']:.1%}, "
              f"Multi-move: {best_results['skills']['multi_move']:.1%}")
        
        # Performance vs baselines
        print(f"\n📊 Performance vs Baselines:")
        for opponent, results in best_results.items():
            if opponent in ['skills', 'overall_strength']:
                continue
            print(f"   vs {opponent:15}: {results['win_rate']:5.1%} wins")
        
        # Training recommendations
        print(f"\n🎯 Training Recommendations:")
        
        # Check if performance degraded over time
        if len(checkpoints) > 1:
            episodes = [ep for _, _, ep in checkpoints]
            strengths = [all_results[f"Episode {ep}"]['overall_strength'] 
                        for ep in episodes]
            
            final_strength = strengths[-1]
            max_strength = max(strengths)
            max_episode = episodes[strengths.index(max_strength)]
            
            if max_episode != episodes[-1] and max_strength > final_strength * 1.1:
                print(f"   ⚠️  Performance peaked at Episode {max_episode}, then declined")
                print(f"   💡 Consider shorter training (~{max_episode} episodes)")
                print(f"   💡 Or adjust learning rate/epsilon decay")
            else:
                print(f"   ✅ Training progressed well - final agent is strong")
                print(f"   💡 Could try longer training or different hyperparameters")
        
        # Skill-specific recommendations
        skills = best_results['skills']
        if skills['tactical'] < 0.3:
            print(f"   📚 Low tactical skills - try training vs random opponent")
        if skills['strategic'] < 0.3:
            print(f"   🧠 Low strategic skills - try self-play training")
        if skills['multi_move'] < 0.3:
            print(f"   🎯 Low multi-move skills - try longer lookahead or self-play")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained RL agents")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing agent checkpoints')
    parser.add_argument('--games_per_opponent', type=int, default=100,
                       help='Games to play against each baseline opponent')
    parser.add_argument('--games_per_matchup', type=int, default=50,
                       help='Games per matchup in head-to-head tournament')
    
    args = parser.parse_args()
    
    benchmark = BenchmarkSuite(args.checkpoint_dir)
    benchmark.run_comprehensive_benchmark(
        games_per_opponent=args.games_per_opponent,
        games_per_matchup=args.games_per_matchup
    )


if __name__ == "__main__":
    main() 