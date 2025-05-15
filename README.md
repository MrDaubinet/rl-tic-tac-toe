# Play Tic-Tac-Toe against an RL Agent

## The Game

### Rules
Tic-Tac-Toe is a two player competitive game where players take turns marking spaces with either X's or O's on a 3 x 3 grid. 

### Win condition
The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.

## Algorithms
The algorithms are implemented in the `/rl-agents`. The [README.md](rl-agents/README.md) for more information. Implemented agents include

1. Temporal Difference Learning: is a method of learning from experience to predict future rewards. It is a combination of Monte Carlo methods and Dynamic Programming.

2. Value based: Q-learning is a value based algorithm that uses a Q-table to store the value of each action in each state. The agent uses the Q-table to choose the best action to take in each state.

3. Policy based: PPO is a policy based algorithm that uses a neural network to approximate the policy function. The agent uses the neural network to choose the best action to take in each state.

I've also added a visualisation screen for plotting. The Reports on training these agents are displayed in this Weights & Biases report. 

<br>

* 

* 

* 

## Technologies
Svelte.js, Sapper.js, tailwind.css, Firebase Hosting

## How to run

Install dependencies:
```
npm install
```

Run the app:
```
npm run dev
```

## Technologies
Svelte.js, tailwind.css, Firebase Hosting

## How to run

Install dependencies:
```
npm install
```

Run the app:
```
npm run dev
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.