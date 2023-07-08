from abc import ABC, abstractmethod
from board import Board
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

class Agent(ABC):
    @abstractmethod
    def next_action(self, board: Board):
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

    def encode_state(self, board):
        state = ""
        for row in board._grid:
            state += "".join(map(str, row))
        return state

    def next_action(self, board: Board):
        state = self.encode_state(board)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(board.get_posible_actions())
        else:
            q_values = {action: self.q_table[state][action] for action in board.get_posible_actions()}
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, next_state, reward, done):
        current_q = self.q_table[state][action]
        next_q = np.mean(self.q_table[next_state]) if not done else 0
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state][action] = new_q
        self.epsilon *= self.epsilon_decay  # decay epsilon

    def take_action(self, board, action):
        previous_board = board.clone()  # Clone the board before the action is taken
        success = board.add_tile(action, self.player)
        done = board.is_final()

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

        reward = (line_weight * player_lines
                + potential_win_weight * player_potential_wins
                + sandwich_weight * player_sandwiches
                - line_weight * opponent_lines
                - potential_win_weight * opponent_potential_wins
                - sandwich_weight * opponent_sandwiches)

        if done:
            if board.winner == self.player:
                reward += 500  # Big reward when the agent wins
            elif board.is_full():
                reward -= 500  # Big penalty when the board is full and the agent didn't win
            return reward, done

        elif not success:  # Penalty if the action is invalid
            reward -= 50

        return reward, done
    
    def count_lines(self, board: Board, player: int):
        count = 0
        for i in range(board.heigth):
            for j in range(board.length):
                if board[i][j] == player:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if self.check_line(board, i, j, dx, dy, player):
                            count += 1
        return count
    
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.board_shape[1]), pickle.load(f))

    def plot_rewards(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.show()
    
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

    # def play_episode(agent, opponent):
    #     board = Board(6, 7)
    #     player_turn = 1
    #     total_reward = 0

    #     while not board.is_final():
    #         if player_turn == agent.player:
    #             current_state = agent.encode_state(board)
    #             action = agent.next_action(board)
    #             reward, done = agent.take_action(board, action)
    #             next_state = agent.encode_state(board)

    #             if not done:
    #                 agent.update_q_table(current_state, action, next_state, reward, done)
    #                 total_reward += reward

    #         else:  # opponent's turn
    #             action = opponent.next_action(board)
    #             board.add_tile(action, opponent.player)

    #         player_turn = 3 - player_turn  # switch players

    #     return total_reward

    # def train_agent(agent, opponent, num_episodes):
    #     episode_rewards = []

    #     for episode in range(num_episodes):
    #         episode_reward = play_episode(agent, opponent)
    #         episode_rewards.append(episode_reward)

    #         if (episode + 1) % 100 == 0:
    #             print(f"Episode {episode + 1}: Total reward = {episode_reward}")

    #     return episode_rewards
