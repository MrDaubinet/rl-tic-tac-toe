# Tic Tac Toe RL Web App

This is a modern SvelteKit frontend for playing Tic Tac Toe against reinforcement learning agents trained with Temporal Difference (TD), Q-Learning, and MinMaxTD algorithms.

## Features
- Play against different RL agents (TD, QLearning, MinMaxTD)
- Switch agent type and agent role (X or O) in the settings panel
- Visualize the agent's value function for each possible move
- Responsive, accessible UI with game history, undo, and reset

## Getting Started

### 1. Install dependencies
```bash
cd web-app
npm install
```

### 2. Run the development server
```bash
npm run dev
```
- The app will be available at `http://localhost:5173` (or as indicated in the terminal).

### 3. Switch agents and settings
- Click the settings button (top right) to open the settings panel.
- Choose the agent type (TD, QLearning, MinMaxTD) and which player is controlled by the agent (X or O).
- Toggle value function display to see the agent's evaluation for each possible move.

### 4. Play
- Click on the board to make your move. The agent will respond automatically if enabled.
- Use the reset button to start a new game.
- View game history and undo moves as needed.

## Notes
- Agent weights are loaded from the `static/` directory. Make sure the corresponding `.json` files are present for each agent type and role.
- For best results, use agents trained with sufficient episodes and lookahead depth (see the main project README for training details).

---
For backend training and more details, see the main `reinforcement-learning/README.md`.
