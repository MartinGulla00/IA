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
    
    @abstractmethod
    def heuristic_utility(self, board: Board):
        pass

# no anda
class QLearningAgent(Agent):
    def __init__(self, board_shape, player, learning_rate=0.1, discount_factor=0.9, initial_epsilon=0.9, epsilon_decay=0.999):
        self.player = player
        self.opponent = 3 - player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros(board_shape + (7,))
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.next_action = None
        self.episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_epsilon = []
        self.episode_win = []
        self.episode_draw = []
        self.episode_loss = []
        self.episode_winrate = []
        self.episode_drawrate = []
        self.episode_lossrate = []

    def next_action(self, board):
        self.next_state = board
        self.next_action = self.get_action(board)
        return self.next_action
    
    def get_action(self, board):
        if np.random.random() < self.epsilon:
            return np.random.choice(board.get_posible_actions())
        else:
            return np.argmax(self.q_table[board])
        
    def heuristic_utility(self, board: Board):
        return 0
    
    def update(self):
        if self.state is not None:
            self.q_table[self.state][self.action] += self.learning_rate * (self.reward + self.discount_factor * np.max(self.q_table[self.next_state]) - self.q_table[self.state][self.action])
        self.state = self.next_state
        self.action = self.next_action

    def end_episode(self, reward, win, draw, loss):
        self.episode += 1
        self.episode_rewards.append(reward)
        self.episode_lengths.append(len(self.episode_rewards))
        self.episode_epsilon.append(self.epsilon)
        self.episode_win.append(win)
        self.episode_draw.append(draw)
        self.episode_loss.append(loss)
        self.episode_winrate.append(sum(self.episode_win) / self.episode)
        self.episode_drawrate.append(sum(self.episode_draw) / self.episode)
        self.episode_lossrate.append(sum(self.episode_loss) / self.episode)
        self.epsilon *= self.epsilon_decay
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.next_action = None

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def plot(self):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_lengths, label='Episode length')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_rewards, label='Episode reward')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_epsilon, label='Epsilon')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_winrate, label='Win rate')
        plt.plot(self.episode_drawrate, label='Draw rate')
        plt.plot(self.episode_lossrate, label='Loss rate')
        plt.legend()
        plt.show()
    
    def plot_winrate(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.episode_winrate, label='Win rate')
        plt.plot(self.episode_drawrate, label='Draw rate')
        plt.plot(self.episode_lossrate, label='Loss rate')
        plt.legend()
        plt.show()

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
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
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
    
class MinimaxAgentConAlphaBetaPruning(Agent):
    def __init__(self, player, max_depth=3):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth

    def next_action(self, board):
        action, _ = self.minimax(board, self.player, self.max_depth, float('-inf'), float('inf'))
        
        # Add a small level of randomness to the action selection
        random_factor = 0.1
        if random.random() < random_factor:
            actions = board.get_posible_actions()
            action = random.choice(actions)
        
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
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
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

class MinimaxAgentMejorado(Agent):
    def __init__(self, player, max_depth=3):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.transposition_table = {}  # Initialize the transposition table

    def next_action(self, board):
        action, _ = self.minimax(board, self.player, self.max_depth, float('-inf'), float('inf'))
        
        # Add a small level of randomness to the action selection
        random_factor = 0.1
        if random.random() < random_factor:
            actions = board.get_posible_actions()
            action = random.choice(actions)
        
        return action

    def minimax(self, board: Board, player, depth, alpha, beta):
        if depth == 0 or board.is_final():
            return None, self.heuristic_utility(board)

        state_key = self.encode_state(board)
        if state_key in self.transposition_table:  # Check if the state has already been evaluated
            return self.transposition_table[state_key]

        actions = board.get_posible_actions()
        actions.sort(key=lambda x: abs(x - board.length // 2))  # Sort actions based on their distance to the center column

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

        self.transposition_table[state_key] = (best_action, best_value)  # Store the result in the transposition table
        return best_action, best_value
    
    def encode_state(self, board):
        state = ""
        for row in board._grid:
            state += "".join(map(str, row))
        return state
    
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
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
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
    
class ExpectimaxAgent(Agent):
    def __init__(self, player, max_depth=3):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth

    def next_action(self, board):
        action, _ = self.expectimax(board, self.player, self.max_depth)
        return action

    def expectimax(self, board: Board, player, depth):
        if depth == 0 or board.is_final():
            return None, self.heuristic_utility(board)

        actions = board.get_posible_actions()
        if player == self.player:  # Maximizing player
            best_value = float('-inf')
            best_action = None
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.expectimax(child, self.opponent, depth - 1)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value
        else:  # Averaging player (opponent)
            total_value = 0
            for action in actions:
                child = board.clone()
                child.add_tile(action, player)
                _, value = self.expectimax(child, self.player, depth - 1)
                total_value += value
            avg_value = total_value / len(actions) if actions else 0
            return None, avg_value

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
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if self.check_sandwich(board, i, j, dx, dy, player):
                            count += 1
        return count

    def check_line(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        for _ in range(3):  # Change this to suit the number of tokens required for a line.
            x += dx
            y += dy
            if not self.in_board(board, x, y) or board[x][y] != player:
                return False
        return True

    def check_potential_win(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        count = 0
        for _ in range(3):  # Change this to suit the number of tokens required for a potential win.
            x += dx
            y += dy
            if not self.in_board(board, x, y) or board[x][y] not in [0, player]:
                return False
            if board[x][y] == player:
                count += 1
        return count == 2  # A potential win requires two of the player's tokens and one empty space.

    def check_sandwich(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        x += dx
        y += dy
        if not self.in_board(board, x, y) or board[x][y] not in [0, 3 - player]:
            return False
        x += dx
        y += dy
        if not self.in_board(board, x, y) or board[x][y] != player:
            return False
        return True  # A sandwich requires one of the opponent's tokens between two of the player's tokens.

    def in_board(self, board: Board, x: int, y: int):
        return 0 <= x < board.heigth and 0 <= y < board.length


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