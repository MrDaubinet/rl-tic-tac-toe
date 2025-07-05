// Base agent interface
export interface BaseAgent {
  player: 'X' | 'O';
  chooseAction(board: number[]): number;
}

// Abstract base class for agents
export abstract class Agent implements BaseAgent {
  player: 'X' | 'O';

  constructor(player: 'X' | 'O') {
    this.player = player;
  }

  // Convert board array to string key (match Python format)
  getStateKey(board: number[]): string {
    return board.map(v => v === 0 ? '-' : (v === 1 ? 'X' : 'O')).join('');
  }

  // Get valid moves
  getValidMoves(board: number[]): number[] {
    return board.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
  }

  abstract chooseAction(board: number[]): number;
}

// TD Learning Agent
export class TDAgent extends Agent {
  valueFunction: Record<string, number>;

  constructor(valueFunction: Record<string, number>, player: 'X' | 'O') {
    super(player);
    this.valueFunction = valueFunction;
  }

  getValue(stateKey: string): number {
    return this.valueFunction[stateKey] ?? 0.1; // Default optimistic value
  }

  chooseAction(board: number[]): number {
    const validMoves = this.getValidMoves(board);
    if (validMoves.length === 0) return -1;

    let bestValue = -Infinity;
    let bestMoves: number[] = [];

    for (const move of validMoves) {
      const nextBoard = [...board];
      nextBoard[move] = this.player === 'X' ? 1 : 2;
      const stateKey = this.getStateKey(nextBoard);
      const value = this.getValue(stateKey);

      if (value > bestValue) {
        bestValue = value;
        bestMoves = [move];
      } else if (value === bestValue) {
        bestMoves.push(move);
      }
    }

    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  static async loadFromUrl(url: string, player: 'X' | 'O'): Promise<TDAgent> {
    const res = await fetch(url);
    const valueFunction = await res.json();
    return new TDAgent(valueFunction, player);
  }
}

// LookAhead Agent (uses TD values + minimax)
export class MinMaxTDAgent extends Agent {
  valueFunction: Record<string, number>;
  lookaheadDepth: number;

  constructor(valueFunction: Record<string, number>, player: 'X' | 'O', lookaheadDepth: number = 2) {
    super(player);
    this.valueFunction = valueFunction;
    this.lookaheadDepth = lookaheadDepth;
  }

  getValue(stateKey: string): number {
    return this.valueFunction[stateKey] ?? 0.1;
  }

  // Check if game is over and return terminal value
  evaluateTerminalState(board: number[]): number | null {
    const lines = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6] // Diagonals
    ];

    const myMarker = this.player === 'X' ? 1 : 2;
    const opponentMarker = this.player === 'X' ? 2 : 1;

    // Check if we won
    for (const line of lines) {
      if (line.every(i => board[i] === myMarker)) {
        return 1.0; // We win
      }
    }

    // Check if opponent won
    for (const line of lines) {
      if (line.every(i => board[i] === opponentMarker)) {
        return -1.0; // Opponent wins
      }
    }

    // Check for draw
    if (board.every(cell => cell !== 0)) {
      return 0.0; // Draw
    }

    return null; // Not terminal
  }

  minimax(board: number[], depth: number, isMaximizing: boolean, alpha: number = -Infinity, beta: number = Infinity): number {
    const terminalValue = this.evaluateTerminalState(board);
    if (terminalValue !== null || depth === 0) {
      if (terminalValue !== null) {
        return terminalValue;
      }
      // Use learned value function for non-terminal positions
      const stateKey = this.getStateKey(board);
      return this.getValue(stateKey);
    }

    const validMoves = this.getValidMoves(board);
    const currentPlayer = isMaximizing ? (this.player === 'X' ? 1 : 2) : (this.player === 'X' ? 2 : 1);

    if (isMaximizing) {
      let maxValue = -Infinity;
      for (const move of validMoves) {
        const nextBoard = [...board];
        nextBoard[move] = currentPlayer;
        const value = this.minimax(nextBoard, depth - 1, false, alpha, beta);
        maxValue = Math.max(maxValue, value);
        alpha = Math.max(alpha, value);
        if (beta <= alpha) break; // Alpha-beta pruning
      }
      return maxValue;
    } else {
      let minValue = Infinity;
      for (const move of validMoves) {
        const nextBoard = [...board];
        nextBoard[move] = currentPlayer;
        const value = this.minimax(nextBoard, depth - 1, true, alpha, beta);
        minValue = Math.min(minValue, value);
        beta = Math.min(beta, value);
        if (beta <= alpha) break; // Alpha-beta pruning
      }
      return minValue;
    }
  }

  chooseAction(board: number[]): number {
    const validMoves = this.getValidMoves(board);
    if (validMoves.length === 0) return -1;

    let bestValue = -Infinity;
    let bestMoves: number[] = [];

    for (const move of validMoves) {
      const nextBoard = [...board];
      nextBoard[move] = this.player === 'X' ? 1 : 2;
      const value = this.minimax(nextBoard, this.lookaheadDepth - 1, false);

      if (value > bestValue) {
        bestValue = value;
        bestMoves = [move];
      } else if (value === bestValue) {
        bestMoves.push(move);
      }
    }

    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  static async loadFromUrl(url: string, player: 'X' | 'O'): Promise<MinMaxTDAgent> {
    const res = await fetch(url);
    const valueFunction = await res.json();
    return new MinMaxTDAgent(valueFunction, player);
  }
}

// Q-Learning Agent
export class QLearningAgent extends Agent {
  qFunction: Record<string, Record<string, number>>;

  constructor(qFunction: Record<string, Record<string, number>>, player: 'X' | 'O') {
    super(player);
    this.qFunction = qFunction;
  }

  getActionValue(stateKey: string, action: number): number {
    return this.qFunction[stateKey]?.[action.toString()] ?? 0.0;
  }

  getMaxActionValue(stateKey: string, validActions: number[]): number {
    let maxValue = -Infinity;
    for (const action of validActions) {
      const value = this.getActionValue(stateKey, action);
      maxValue = Math.max(maxValue, value);
    }
    return maxValue === -Infinity ? 0.0 : maxValue;
  }

  chooseAction(board: number[]): number {
    const validMoves = this.getValidMoves(board);
    if (validMoves.length === 0) return -1;

    const stateKey = this.getStateKey(board);
    let bestValue = -Infinity;
    let bestMoves: number[] = [];

    for (const move of validMoves) {
      const value = this.getActionValue(stateKey, move);
      if (value > bestValue) {
        bestValue = value;
        bestMoves = [move];
      } else if (value === bestValue) {
        bestMoves.push(move);
      }
    }

    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  static async loadFromUrl(url: string, player: 'X' | 'O'): Promise<QLearningAgent> {
    const res = await fetch(url);
    const qFunction = await res.json();
    return new QLearningAgent(qFunction, player);
  }
}

// Agent factory function
export type AgentType = 'TDAgent' | 'MinMaxTDAgent' | 'QLearning';

export class AgentFactory {
  static async loadAgent(type: AgentType, player: 'X' | 'O'): Promise<BaseAgent> {
    const url = `/${type}_${player}.json`;
    
    switch (type) {
      case 'TDAgent':
        return await TDAgent.loadFromUrl(url, player);
      case 'MinMaxTDAgent':
        return await MinMaxTDAgent.loadFromUrl(url, player);
      case 'QLearning':
        return await QLearningAgent.loadFromUrl(url, player);
      default:
        throw new Error(`Unknown agent type: ${type}`);
    }
  }

  static getAvailableAgentTypes(): AgentType[] {
    return ['TDAgent', 'QLearning', 'MinMaxTDAgent'];
  }
} 