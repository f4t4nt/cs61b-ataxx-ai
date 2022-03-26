import numpy as np
import random as rd

#  1: player 1 to move
#  0: empty
# -1: player 2
# -2: wall
class AtaxxBoard():

    def __init__(self, side_length = 7, jump_limit = 25, wall_p = 0.2, board = np.zeros(1)):
        if np.all(board == np.zeros(1)):
            self.SIDE_LENGTH = side_length
            self.SIDE_LENGTH_EXTENDED = side_length + 4
            self.JUMP_LIMIT = jump_limit
            self.reset(wall_p)
        else:
            self.SIDE_LENGTH = board.shape[0] - 4
            self.SIDE_LENGTH_EXTENDED = board.shape[0]
            self.JUMP_LIMIT = jump_limit
            self.walls = abs(board) // 2 * -2
            self.pieces = board - self.walls
            self.board = board

    def get_board(self):
        self.board = self.pieces + self.walls
        return self.board

    def reset(self, wall_p):
        self.pieces = np.zeros([self.SIDE_LENGTH_EXTENDED, self.SIDE_LENGTH_EXTENDED])
        self.pieces[2, 2] = -1
        self.pieces[self.SIDE_LENGTH + 1, 2] = 1
        self.pieces[2, self.SIDE_LENGTH + 1] = 1
        self.pieces[self.SIDE_LENGTH + 1, self.SIDE_LENGTH + 1] = -1
        self.walls = np.zeros([self.SIDE_LENGTH_EXTENDED, self.SIDE_LENGTH_EXTENDED]) - 2
        for col in range((self.SIDE_LENGTH + 1) // 2):
            for row in range((self.SIDE_LENGTH + 1) // 2):
                if (col == 0 and row == 0) or rd.random() > wall_p:
                    col0, row0 = col + 2, row + 2
                    col1, row1 = self.SIDE_LENGTH + 1 - col, self.SIDE_LENGTH + 1 - row
                    self.walls[col0, row0] = 0
                    self.walls[col0, row1] = 0
                    self.walls[col1, row0] = 0
                    self.walls[col1, row1] = 0
        self.get_board()
        self.num_jumps = 0

    def get_winner(self):
        if self.num_jumps == self.JUMP_LIMIT:
            return (0, True)
        p1_pieces = np.sum(self.pieces == 1)
        p2_pieces = np.sum(self.pieces == -1)
        if p1_pieces == 0:
            return (-1, True)
        if p2_pieces == 0:
            return (1, True)
        if len(self.get_moves(-1)) > 0 or len(self.get_moves(1)) > 0:
            return (0, False)
        if p1_pieces > p2_pieces:
            return (1, True)
        if p1_pieces < p2_pieces:
            return (-1, True)
        if p1_pieces == p2_pieces:
            return (0, True)
        
    def get_moves(self, player):
        self.get_board()
        moves = []
        for col in range(2, self.SIDE_LENGTH + 2):
            for row in range(2, self.SIDE_LENGTH + 2):
                if self.board[col, row] == player:
                    for dc in range(-2, 3):
                        for dr in range(-2, 3):
                            if self.board[col + dc, row + dr] == 0:
                                moves.append((col, row, col + dc, row + dr))
        if len(moves) == 0:
            moves.append((0, 0, 0, 0))
        return moves

    def make_move(self, player, move):
        assert player == 1
        assert move in self.get_moves(player)
        if move != (0, 0, 0, 0):
            col0, row0, col1, row1 = move
            if max(abs(col0 - col1), abs(row0 - row1)) > 1:
                self.num_jumps += 1
                self.pieces[col0, row0] = 0
            self.pieces[col1, row1] = player
            self.flip_pieces(col1, row1, player)

    def flip_pieces(self, col, row, player):
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                if self.pieces[col + dc, row + dr] == -player:
                    self.pieces[col + dc, row + dr] = player

    def to_string(self):
        board_str = ""
        for col in range(2, self.SIDE_LENGTH + 2):
            for row in range(2, self.SIDE_LENGTH, + 2):
                board_str += str(self.board[col, row])
            board_str += "\n"
        return board_str

    def to_string_readable(self):
        self.get_board()
        board_str = ""
        for row in range(self.SIDE_LENGTH + 1, 1, -1):
            board_str += str(row - 1)
            for _ in range(len(str(row - 1)), len(str(self.SIDE_LENGTH + 1))):
                board_str += " "
            for col in range(2, self.SIDE_LENGTH + 2):
                board_str += " "
                if self.board[col, row] == 1:
                    board_str += "●"
                elif self.board[col, row] == -1:
                    board_str += "○"
                elif self.board[col, row] == 0:
                    board_str += "-"
                else:
                    board_str += "■"
            board_str += "\n"
        for _ in range(len(str(self.SIDE_LENGTH + 1))):
            board_str += " "
        for col in range(self.SIDE_LENGTH):
            board_str += " " + chr(ord('a') + col)
        board_str += "\n"
        return board_str