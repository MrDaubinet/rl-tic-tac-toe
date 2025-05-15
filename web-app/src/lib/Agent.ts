export class TDAgent {
  valueFunction: Record<string, number>;
  player: 'X' | 'O';

  constructor(valueFunction: Record<string, number>, player: 'X' | 'O') {
    this.valueFunction = valueFunction;
    this.player = player;
  }

  // Convert board array to string key (match Python)
  getStateKey(board: number[]): string {
    // 0: empty, 1: X, 2: O
    return board.map(v => v === 0 ? '-' : (v === 1 ? 'X' : 'O')).join('');
  }

  // Get valid moves
  getValidMoves(board: number[]): number[] {
    return board.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
  }

  // Choose the best move
  chooseAction(board: number[]): number {
    const validMoves = this.getValidMoves(board);
    let bestValue = -Infinity;
    let bestMoves: number[] = [];
    for (const move of validMoves) {
      const nextBoard = [...board];
      nextBoard[move] = this.player === 'X' ? 1 : 2;
      const stateKey = this.getStateKey(nextBoard);
      const value = this.valueFunction[stateKey] ?? 0;
      if (value > bestValue) {
        bestValue = value;
        bestMoves = [move];
      } else if (value === bestValue) {
        bestMoves.push(move);
      }
    }
    // Randomly choose among the best moves
    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  // Static async loader
  static async loadFromUrl(url: string, player: 'X' | 'O'): Promise<TDAgent> {
    const res = await fetch(url);
    const valueFunction = await res.json();
    return new TDAgent(valueFunction, player);
  }
} 