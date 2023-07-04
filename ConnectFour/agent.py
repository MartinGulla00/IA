from abc import ABC, abstractmethod
from board import Board
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Agent(ABC):

    @abstractmethod
    def next_action(self, obs):
        return NotImplementedError
    
    @abstractmethod
    def heuristic_utility(self, board: Board):
        return NotImplementedError

from abc import ABC, abstractmethod
from board import Board
import numpy as np

class Agent(ABC):

    @abstractmethod
    def next_action(self, obs):
        pass
    
    @abstractmethod
    def heuristic_utility(self, board: Board):
        pass

class QLearningAgent(Agent):
    def __init__(self, board_shape, player, learning_rate=0.1, discount_factor=0.9, initial_epsilon=0.9, epsilon_decay=0.999):
        self.board_shape = board_shape
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(board_shape[1]))
        self.episode_rewards = []  # <--- Agregado aquí

    def encode_state(self, board):
        state = ""
        for row in board._grid:
            state += "".join(map(str, row))
        return state

    def next_action(self, obs):
        state = self.encode_state(obs)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(obs.get_posible_actions())
        else:
            q_values = {action: self.q_table[state][action] for action in obs.get_posible_actions()}
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, next_state, reward, done):
        current_q = self.q_table[state][action]
        next_q = np.mean(self.q_table[next_state]) if not done else 0
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state][action] = new_q
        self.epsilon *= self.epsilon_decay  # decay epsilon


    def train(self, num_episodes, verbose=False, plot=True):
        for episode in range(num_episodes):
            obs = Board(self.board_shape[0], self.board_shape[1])  # Create a new board for each episode
            done = False
            episode_reward = 0  # Track the total reward for the episode
            iteration_count = 0  # Track the number of iterations for each episode
            while not done:
                iteration_count += 1
                current_state = self.encode_state(obs)
                action = self.next_action(obs)
                reward, done = self.take_action(obs, action)
                obs.add_tile(action, self.player)  # <-- Aquí se actualiza obs
                next_state = self.encode_state(obs)
                self.update_q_table(current_state, action, next_state, reward, done)
                episode_reward += reward
            self.episode_rewards.append(episode_reward)
            # At the end of each episode, print out the total number of iterations
            if verbose:
                print(f"Episode {episode + 1} completed after {iteration_count} iterations.")

        if plot:
            self.plot_rewards()


    def plot_rewards(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.show()

    def take_action(self, obs, action):
        previous_board = obs.clone()  # Clone the board before the action is taken
        success = obs.add_tile(action, self.player)
        done = obs.is_final()

        player = self.player
        opponent = 3 - player  # Assumed that players are 1 and 2

        player_lines = self.count_lines(obs, player)
        opponent_lines = self.count_lines(obs, opponent)
        
        player_potential_wins = self.count_potential_wins(obs, player)
        opponent_potential_wins = self.count_potential_wins(obs, opponent)

        player_sandwiches = self.count_sandwiches(obs, player)
        opponent_sandwiches = self.count_sandwiches(obs, opponent)

        # These weights can be tuned according to the importance you want to give each factor.
        line_weight = 10
        potential_win_weight = 100
        sandwich_weight = 5

        reward = (line_weight * player_lines
                + potential_win_weight * player_potential_wins
                + sandwich_weight * player_sandwiches
                - line_weight * opponent_lines
                - potential_win_weight * opponent_potential_wins
                - sandwich_weight * opponent_sandwiches)

        if done:
            if obs.winner == self.player:
                reward += 500  # Big reward when the agent wins
            elif obs.is_full():
                reward -= 500  # Big penalty when the board is full and the agent didn't win
            return reward, done  # <-- Modificado aquí

        elif not success:  # Penalty if the action is invalid
            reward -= 50

        return reward, done


    def heuristic_utility(self, board: Board):
        utility = 0
        opponent = 2 if self.player == 1 else 1

        for row in board._grid:
            for i in range(board._length - 3):
                if (row[i] == row[i+3] == self.player) and (row[i+1] == row[i+2] == opponent):
                    utility += 1
                elif (row[i] == row[i+3] == opponent) and (row[i+1] == row[i+2] == self.player):
                    utility -= 1

        for i in range(board._heigth - 3):
            for j in range(board._length - 3):
                if (board._grid[i][j] == board._grid[i+3][j+3] == self.player) and (board._grid[i+1][j+1] == board._grid[i+2][j+2] == opponent):
                    utility += 1
                elif (board._grid[i][j] == board._grid[i+3][j+3] == opponent) and (board._grid[i+1][j+1] == board._grid[i+2][j+2] == self.player):
                    utility -= 1

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