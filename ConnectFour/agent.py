from abc import ABC, abstractmethod
from board import Board
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import random

class Agent(ABC):

    @abstractmethod
    def next_action(self, obs):
        pass

    def heuristic_utility(self, board: Board):
        player = self.player
        opponent = 3 - player  # Assumed that players are 1 and 2

        player_lines = self.count_lines(board, player)
        opponent_lines = self.count_lines(board, opponent)
        
        player_potential_wins = self.count_potential_wins(board, player)
        opponent_potential_wins = self.count_potential_wins(board, opponent)

        player_sandwiches = self.count_sandwiches(board, player)
        opponent_sandwiches = self.count_sandwiches(board, opponent)

        # These weights can be tuned according to the importance you want to give each factor.
        line_weight = 10
        potential_win_weight = 100
        sandwich_weight = 5

        utility = (line_weight * player_lines
                + potential_win_weight * player_potential_wins
                + sandwich_weight * player_sandwiches
                - line_weight * opponent_lines
                - potential_win_weight * opponent_potential_wins
                - sandwich_weight * opponent_sandwiches)
        
        return utility
    
    def count_lines(self, board: Board, player: int):
        count = 0
        for i in range(board.heigth):
            for j in range(board.length):
                if board[i][j] == player:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if self.check_line(board, i, j, dx, dy, player):
                            count += 1
        return count
    
    def count_potential_wins(self, board: Board, player: int):
        count = 0
        for i in range(board.heigth):
            for j in range(board.length):
                if board[i][j] == player:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if self.check_potential_win(board, i, j, dx, dy, player):
                            count += 1
        return count
    
    def count_sandwiches(self, board: Board, player: int):
        count = 0
        for i in range(board.heigth):
            for j in range(board.length):
                if board[i][j] == player:
                    for dx, dy in [(1, 0), (1, 1), (1, -1)]:
                        if self.check_sandwich(board, i, j, dx, dy, player):
                            count += 1
        return count
    
    def check_line(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        for _ in range(3):
            x += dx
            y += dy
            if not self.in_board(board, x, y) or board[x][y] != player:
                return False
        return True
    
    def check_potential_win(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        count = 0
        for _ in range(3):
            x += dx
            y += dy
            if not self.in_board(board, x, y) or board[x][y] not in [0, player]:
                return False
            if board[x][y] == player:
                count += 1
        return count == 2
    
    def check_sandwich(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        if dx == 1 and dy == 0:  # Ignore vertical sandwiches
            return False

        x += dx
        y += dy
        if not self.in_board(board, x, y) or board[x][y] not in [0, 3 - player]:
            return False
        x += dx
        y += dy
        if not self.in_board(board, x, y) or board[x][y] != player:
            return False
        return True
    
    def in_board(self, board: Board, x: int, y: int):
        return 0 <= x < board.heigth and 0 <= y < board.length

class MinimaxAgent(Agent):
    def __init__(self, player, max_depth=3):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth

    def next_action(self, board):
        action, _ = self.minimax(board, self.player, self.max_depth)
        return action

    def minimax(self, board: Board, player, depth):
        if depth == 0 or board.is_final():
            return None, self.heuristic_utility(board)

        actions = board.get_posible_actions()
        if player == self.player:  # Maximizing player
            best_value = float('-inf')
            best_action = None
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.minimax(child, self.opponent, depth - 1)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value
        else:  # Minimizing player (opponent)
            best_value = float('inf')
            best_action = None
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.minimax(child, self.player, depth - 1)
                if value < best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value
    
class MinimaxAgentConAlphaBetaPruning(Agent):
    def __init__(self, player, max_depth=3):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth

    def next_action(self, board):
        # Verificar si existe una jugada ganadora y realizarla
        for action in board.get_posible_actions():
            child = board.clone()
            child.add_tile(action, self.player)
            if child.is_final():
                return action
        action, _ = self.minimax(board, self.player, self.max_depth, float('-inf'), float('inf'))
        
        return action

    def minimax(self, board: Board, player, depth, alpha, beta):
        if depth == 0 or board.is_final():
            return None, self.heuristic_utility(board)

        actions = board.get_posible_actions()
        if player == self.player:  # Maximizing player
            best_value = float('-inf')
            best_action = None
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.minimax(child, self.opponent, depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            return best_action, best_value
        else:  # Minimizing player (opponent)
            best_value = float('inf')
            best_action = None
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.minimax(child, self.player, depth - 1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_action, best_value

        
class ExpectimaxAgentMejorado(Agent):
    def __init__(self, player, max_depth=3):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.transposition_table = {}  # Initialize the transposition table

    def next_action(self, board):
        # Verificar si existe una jugada ganadora y realizarla
        for action in board.get_posible_actions():
            child = board.clone()
            child.add_tile(action, self.player)
            if child.is_final():
                return action
        action, _ = self.expectimax(board, self.player, self.max_depth, float('-inf'), float('inf'))
        return action
    
    def expectimax(self, board: Board, player, depth, alpha, beta):
        if depth == 0 or board.is_final():
            return None, self.heuristic_utility(board)
        
        state_key = self.encode_state(board)
        if state_key in self.transposition_table:  # Check if the state has already been evaluated
            return self.transposition_table[state_key]
        
        actions = board.get_posible_actions()
        best_value = float('-inf')
        best_action = None

        if player == self.player:  # Maximizing player
            best_value = float('-inf')
            best_action = None
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.expectimax(child, self.opponent, depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                if value == best_value and random.random() < 0.5:
                    best_action = action
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
        else:  # Chance player (opponent)
            value = 0
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, child_value = self.expectimax(child, self.player, depth - 1, alpha, beta)
                value += child_value
            value /= len(actions)
            best_value = value
            best_action = None
        
        self.transposition_table[state_key] = (best_action, best_value)  # Store the result in the transposition table
        return best_action, best_value
        
    def encode_state(self, board):
        state = ""
        for row in board._grid:
            state += "".join(map(str, row))
        return state
    

class InputAgent(Agent):

    def next_action(self, obs):
        while True:
            try:
                input_value = int(input())
                return input_value
            except ValueError:
                print("Please insert a number.")

    def heuristic_utility(self, board: Board):
        return 0