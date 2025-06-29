# Reinforcement Learning Visualization Module

## Algorithms

1. Temporal Difference Learning: is a method of learning from experience to predict future rewards. It is a combination of Monte Carlo methods and Dynamic Programming.

2. Value based: Q-learning is a value based algorithm that uses a Q-table to store the value of each action in each state. The agent uses the Q-table to choose the best action to take in each state.

3. Policy based: PPO is a policy based algorithm that uses a neural network to approximate the policy function. The agent uses the neural network to choose the best action to take in each state.

## Overview
This module provides comprehensive visualization tools for tracking and analyzing the performance of reinforcement learning algorithms, specifically designed for the Tic-Tac-Toe learning environment. It offers intuitive visualizations of learning progress, win rates, rewards, and value functions.

## Key Features
- Real-time tracking of learning metrics
- Interactive learning curve plots
- Value function heatmap visualization
- Customizable plot styling and saving options
- Clear and intuitive API

## Dependencies
- Python 3.7+
- NumPy (>=1.21.0)
- Matplotlib (>=3.4.0)
- Seaborn (>=0.11.0)

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/rl-tic-tac-toe.git
cd rl-tic-tac-toe
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Basic Example
```python
from visualization import LearningVisualizer

# Initialize the visualizer
visualizer = LearningVisualizer()

# Update metrics during training
visualizer.update_win_rate(episode=100, win_rate=0.65)
visualizer.update_rewards(episode=100, reward=0.8)

# Plot learning curves
visualizer.plot_learning_curves(save_path='learning_curves.png')

# Visualize value function
board_state = [0, 1, -1, 0, 0, 0, 1, -1, 0]  # Example board state
visualizer.plot_value_function_heatmap(board_state, save_path='value_function.png')
```

## Types of Visualizations

### 1. Learning Curves
- Win rate over episodes
- Rewards over episodes
- Customizable plotting options
- Support for saving plots to files

### 2. Value Function Heatmaps
- 3x3 grid visualization of state values
- Color-coded value representation
- Intuitive interpretation of learned strategies
- Support for any valid board state

## Types of agents implemented


