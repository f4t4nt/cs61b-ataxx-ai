import numpy as np
import random as rd

#  1: red
#  0: empty
# -1: blue
# -2: wall
class Board():

    def __init__(self, side_length = 7, jump_limit = 25, wall_p = 0.2):
        self.SIDE_LENGTH = side_length
        self.JUMP_LIMIT = jump_limit
        self.reset(wall_p)
        
    def setToBoard(self, board, jump_limit = 25):
        self.SIDE_LENGTH = board.shape[0]
        self.JUMP_LIMIT = jump_limit
        self.walls = abs(board) // 2 * -2
        self.pieces = board - self.walls
        self.board = board
        return self

    def getBoard(self):
        self.board = self.pieces + self.walls
        return self.board

    def reset(self, wall_p):
        self.pieces = np.zeros([self.SIDE_LENGTH, self.SIDE_LENGTH])
        self.pieces[0, 0] = -1
        self.pieces[self.SIDE_LENGTH - 1, 0] = 1
        self.pieces[0, self.SIDE_LENGTH - 1] = 1
        self.pieces[self.SIDE_LENGTH - 1, self.SIDE_LENGTH - 1] = -1
        self.walls = np.zeros([self.SIDE_LENGTH, self.SIDE_LENGTH])
        for col0 in range((self.SIDE_LENGTH + 1) // 2):
            for row0 in range((self.SIDE_LENGTH + 1) // 2):
                if (col0 > 0 or row0 > 0) and rd.random() < wall_p:
                    col1, row1 = self.SIDE_LENGTH - 1 - col0, self.SIDE_LENGTH - 1 - row0
                    self.walls[col0, row0] = -2
                    self.walls[col0, row1] = -2
                    self.walls[col1, row0] = -2
                    self.walls[col1, row1] = -2
        self.num_jumps = 0
        self.getBoard()

    def getWinner(self):
        if self.num_jumps == self.JUMP_LIMIT:
            return (0, True)
        self.getBoard()
        p1_pieces = np.sum(self.board == 1)
        p2_pieces = np.sum(self.board == -1)
        if p1_pieces == 0:
            return (-1, True)
        if p2_pieces == 0:
            return (1, True)
        if self.canMove(1) or self.canMove(-1):
            return (0, False)
        if p1_pieces > p2_pieces:
            return (1, True)
        if p1_pieces < p2_pieces:
            return (-1, True)
        if p1_pieces == p2_pieces:
            return (0, True)
        
    def getMoves(self, player):
        self.getBoard()
        moves = []
        for col0 in range(self.SIDE_LENGTH):
            for row0 in range(self.SIDE_LENGTH):
                if self.board[col0, row0] == player:
                    for col1 in range(max(0, col0 - 2), min(self.SIDE_LENGTH, col0 + 3)):
                        for row1 in range(max(0, row0 - 2), min(self.SIDE_LENGTH, row0 + 3)):
                            if self.board[col1, row1] == 0:
                                moves.append((col0, row0, col1, row1))
        # for col in range(self.SIDE_LENGTH):
        #     for row in range(self.SIDE_LENGTH):
        #         if self.board[col, row] == player:
        #             for dc in range(-2, 3):
        #                 for dr in range(-2, 3):
        #                     if col + dc >= 0 and \
        #                         col + dc < self.SIDE_LENGTH and \
        #                         row + dr >= 0 and \
        #                         row + dr < self.SIDE_LENGTH and \
        #                         self.board[col + dc, row + dr] == 0:
        #                         moves.append((col, row, col + dc, row + dr))
        if len(moves) == 0:
            moves.append((0, 0, 0, 0))
        return moves

    def canMove(self, player):
        moves = self.getMoves(player)
        return len(moves) > 1 or not moves[0] == (0, 0, 0, 0)

    def makeMove(self, player, move):
        assert move in self.getMoves(player)
        if move != (0, 0, 0, 0):
            col0, row0, col1, row1 = move
            if max(abs(col0 - col1), abs(row0 - row1)) > 1:
                self.num_jumps += 1
                self.pieces[col0, row0] = 0
            else:
                self.num_jumps = 0
            self.pieces[col1, row1] = player
            self.flipPieces(col1, row1, player)
        self.getBoard()

    def flipPieces(self, col1, row1, player):
        for col in range(max(0, col1 - 1), min(self.SIDE_LENGTH, col1 + 2)):
            for row in range(max(0, row1 - 1), min(self.SIDE_LENGTH, row1 + 2)):
                if self.pieces[col, row] == -player:
                    self.pieces[col, row] = player
        # for dc in range(-1, 2):
        #     for dr in range(-1, 2):
        #         if col + dc >= 0 and \
        #             col + dc < self.SIDE_LENGTH and \
        #             row + dr >= 0 and \
        #             row + dr < self.SIDE_LENGTH and \
        #             self.pieces[col + dc, row + dr] == -player:
        #             self.pieces[col + dc, row + dr] = player

    def toString(self):
        self.getBoard()
        board_str = ''
        for col in range(self.SIDE_LENGTH):
            for row in range(self.SIDE_LENGTH):
                board_str += str(self.board[col, row]) + ' '
            board_str += '\n'
        return board_str

    def toStringReadable(self):
        self.getBoard()
        board_str = ''
        for row in range(self.SIDE_LENGTH, 0, -1):
            board_str += str(row)
            for _ in range(len(str(row - 1)), len(str(self.SIDE_LENGTH + 1))):
                board_str += ' '
            for col in range(self.SIDE_LENGTH):
                board_str += ' '
                if self.board[col, row - 1] == 1:
                    board_str += '●'
                elif self.board[col, row - 1] == -1:
                    board_str += '○'
                elif self.board[col, row - 1] == 0:
                    board_str += '-'
                else:
                    board_str += '■'
            board_str += "\n"
        for _ in range(len(str(self.SIDE_LENGTH + 1))):
            board_str += ' '
        for col in range(self.SIDE_LENGTH):
            board_str += ' ' + chr(ord('a') + col)
        board_str += '\n'
        return board_str

# random play for testing

# while True:
#     game = Board()
#     player = 1
#     print(game.toStringReadable())
#     while not game.getWinner()[1]:
#         game.makeMove(player, game.getMoves(player)[rd.randint(0, len(game.getMoves(player)) - 1)])
#         player = -player
#         print(game.toStringReadable())
#     print(game.getWinner()[0])