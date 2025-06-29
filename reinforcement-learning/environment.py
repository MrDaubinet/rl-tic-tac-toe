from typing import List, Tuple

class TicTacToeEnv:
    """Simple Tic-Tac-Toe environment."""
    
    def __init__(self):
        self.board = ['-'] * 9
        self.current_player = 'X'
    
    def reset(self) -> List[str]:
        """Reset the game board."""
        self.board = ['-'] * 9
        self.current_player = 'X'
        return self.board.copy()
    
    def step(self, action: int) -> Tuple[List[str], float, bool]:
        """
        Make a move on the board.
        
        Args:
            action: Position to place the mark (0-8)
            
        Returns:
            tuple: (new_state, reward, done)
        """
        if self.board[action] != '-':
            return self.board.copy(), -1, True  # Invalid move
        
        self.board[action] = self.current_player
        
        # Check for win
        if self._check_win():
            return self.board.copy(), 1.0, True
        
        # Check for draw
        if '-' not in self.board:
            return self.board.copy(), 0.5, True
        
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self.board.copy(), 0.0, False
    
    def _check_win(self) -> bool:
        """Check if current player has won."""
        # Winning combinations
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for line in lines:
            if all(self.board[i] == self.current_player for i in line):
                return True
        return False