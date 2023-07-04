from abc import ABC, abstractmethod
from board import Board

class Agent(ABC):

    @abstractmethod
    def next_action(self, obs):
        return NotImplementedError
    
    @abstractmethod
    def heuristic_utility(self, board: Board):
        return NotImplementedError



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
    

class ExpectimaxAgent(Agent):
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

        # Si no existe una jugada ganadora, continuar con el expectimax
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
        line_weight = 20
        potential_win_weight = 1000
        potential_lose_weight = 2000
        sandwich_weight = 200

        utility = (line_weight * player_lines
                + potential_win_weight * player_potential_wins
                + sandwich_weight * player_sandwiches
                - line_weight * opponent_lines
                - potential_lose_weight * opponent_potential_wins
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

    def check_line(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        for _ in range(3):  # Change this to suit the number of tokens required for a line.
            x += dx
            y += dy
            if not self.in_board(board, x, y) or board[x][y] != player:
                return False
        return True

    def check_potential_win(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        opponent = 3 - player
        tokens = [board[x + dx * i][y + dy * i] if self.in_board(board, x + dx * i, y + dy * i) else opponent for i in range(4)]
        return (tokens.count(player) == 3 and tokens.count(0) == 1) \
                or (tokens == [player, opponent, player, 0]) \
                or (tokens == [0, opponent, player, player]) \
                or (tokens == [player, 0, player]) # Este es el nuevo caso "X # X"


    def count_sandwiches(self, board: Board, player: int):
        count = 0
        for i in range(board.heigth):
            for j in range(board.length):
                if board[i][j] == player:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if dy != 0:  # Solo se consideran sándwiches en horizontal y diagonal
                            if self.check_sandwich(board, i, j, dx, dy, player):
                                count += 1
        return count

    def check_sandwich(self, board: Board, x: int, y: int, dx: int, dy: int, player: int):
        x += dx
        y += dy
        opponent = 3 - player
        if not self.in_board(board, x, y) or board[x][y] != opponent:
            return False

        # comprobar si hay una pieza del jugador en el otro lado
        x += dx
        y += dy
        if not self.in_board(board, x, y) or board[x][y] != player:
            return False

        return True  # un sándwich requiere una pieza del oponente entre dos piezas del jugador
    def count_potential_wins(self, board: Board, player: int):
        count = 0
        for i in range(board.heigth):
            for j in range(board.length):
                if board[i][j] == player:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if self.check_potential_win(board, i, j, dx, dy, player):
                            count += 1
        return count

    def in_board(self, board: Board, x: int, y: int):
        return 0 <= x < board.heigth and 0 <= y < board.length
