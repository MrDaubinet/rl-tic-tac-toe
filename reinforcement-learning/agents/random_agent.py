import random
from typing import List
from agents.agent import Agent

class RandomAgent(Agent):
    """Agent that makes random valid moves."""
    def choose_action(self, board: List[str], evaluation_mode: bool = False) -> int:
        """Choose a random valid move."""
        valid_moves = self.get_valid_moves(board)
        return random.choice(valid_moves) if valid_moves else -1