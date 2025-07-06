# Reinforcement Learning Visualization & Tic-Tac-Toe Agents

## Algorithms

1. **Temporal Difference Learning (TD):** Learns from experience to predict future rewards, combining Monte Carlo and Dynamic Programming methods.
2. **Value-based (Q-learning):** Uses a Q-table to store the value of each action in each state. The agent uses the Q-table to choose the best action to take in each state.
3. **MinMaxTD Agent:** Combines TD learning with minimax lookahead search for stronger play. Lookahead depth is configurable.
4. **Random Agent:** Selects moves randomly (for benchmarking and as a baseline).

## Overview
This module provides comprehensive tools for training, evaluating, and visualizing reinforcement learning agents for Tic-Tac-Toe. It includes:
- Python backend for agent training and evaluation
- SvelteKit web app frontend for interactive play and value function visualization

## Key Features
- Real-time tracking of learning metrics
- Interactive learning curve plots
- Value function heatmap visualization
- Customizable plot styling and saving options
- Modern, responsive web UI with agent settings and value function overlays
- Play against trained agents directly in the browser
- Toggle between agent types, player roles, and value function display

## Dependencies
- Python 3.7+
- NumPy (>=1.21.0)
- Matplotlib (>=3.4.0)
- Seaborn (>=0.11.0)
- SvelteKit (for frontend)

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
### Basic Example (Python)
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

### Web App (SvelteKit)
- Run the web app in `web-app/` to play against trained agents, visualize value functions, and adjust agent settings interactively.
- Features include:
  - Value function overlay for each possible move
  - Toggle agent type (TD, QLearning, MinMaxTD)
  - Choose agent role (X or O)
  - Responsive, accessible UI
  - Game history and undo
  - Reset and settings panel

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

## Types of Agents Implemented
- TDAgent (Temporal Difference)
- MinMaxTDAgent (TD + Minimax lookahead)
- QLearning Agent
- Random Agent

## Training Options
- **Self-play:** Agent plays against itself
- **Random opponent:** Agent plays against random moves
- **Independent:** Two separate agents (X and O) learn independently
- **TD/MinMax competitor:** Agent plays against a fresh TD or MinMaxTD agent
- **Lookahead depth:** For MinMaxTDAgent, controls how many moves ahead are considered (higher = stronger, slower)

## Best Agent

**My best agent was trained with the following command:**

```bash
python train.py --model minmaxtd --competitor independent --episodes 100000 --learning_rate 0.5 --discount_factor 0.9 --epsilon 1 --decay_rate 0.99995 --min_epsilon 0.10 --lookahead_depth 3
```

- This agent uses MinMaxTD with a lookahead depth of 3, high initial exploration, and independent training for X and O.
- It is robust against most human strategies and can block multi-move traps (like the double-corner fork).

---
For more details, see the code and comments in each module.


