from __future__ import print_function
import sys

from scipy import rand
sys.path.append('..')
from Game import Game
from .AtaxxLogic import Board
import numpy as np


chrs = {
    -2: '#',
    -1: 'O',
    1: 'X',
    2: '#',
    0: ' ',
}

class AtaxxGame(Game):
    def __init__(self, side_length = 7, jump_limit = 25, wall_p = 0.2):
        self.SIDE_LENGTH = side_length
        self.jump_limit = jump_limit
        self.wall_p = wall_p

    def getInitBoard(self):
        self.init_board = Board(self.SIDE_LENGTH, self.jump_limit, self.wall_p)
        return self.init_board.board

    def getBoardSize(self):
        self.board_shape = (self.SIDE_LENGTH, self.SIDE_LENGTH)
        return self.board_shape

    def getActionSize(self):
        self.action_size = 25 * self.SIDE_LENGTH ** 2
        self.idx_denom = 25 * self.SIDE_LENGTH
        return self.action_size

    def moveToidx(self, move):
        (col0, row0, col1, row1) = move
        dr = row1 - row0 + 2
        dc = col1 - col0 + 2
        if dr == dc and dr == 2:
            return 0
        return (col0 * self.SIDE_LENGTH + row0) * 25 + dc * 5 + dr

    def idxToMove(self, move):
        dc = (move % 25) // 5 - 2
        dr = move % 5 - 2
        if dc == 0 and dr == 0:
            col0, row0, col1, row1 = [0] * 4
        else:
            col0 = move // self.idx_denom
            row0 = (move % self.idx_denom) // 25
            col1, row1 = col0 + dc, row0 + dr
        return (col0, row0, col1, row1)

    def boardToClass(self, board):
        return Board().setToBoard(board)

    def getNextState(self, board, player, move):
        board = self.boardToClass(board)
        col0, row0, col1, row1 = self.idxToMove(move)
        board.makeMove(player, (col0, row0, col1, row1))
        return (board.board, -player)
    
    def getValidMoves(self, board, player):
        board = self.boardToClass(board)
        valid_moves = np.zeros(25 * self.SIDE_LENGTH ** 2)
        valid_moves_list = board.getMoves(player)
        for move in range(25 * self.SIDE_LENGTH ** 2):
            col0, row0, col1, row1 = self.idxToMove(move)
            if (col0, row0, col1, row1) in valid_moves_list:
                valid_moves[move] = 1
        return valid_moves
    
    def getGameEnded(self, board, player):
        board = self.boardToClass(board)
        winner, ended = board.getWinner()
        if ended:
            if winner == player:
                return 1
            elif winner == -player:
                return -1
            elif winner == 0:
                return -0.1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        board = self.boardToClass(board)
        return board.pieces * player + board.walls

    def getSymmetries(self, board, pi):
        sym_forms = []
        pi_board = np.reshape(pi, (self.SIDE_LENGTH, self.SIDE_LENGTH, 5, 5))
        for rot in range(4):
            for flip in [False, True]:
                sym_board = np.rot90(board, rot)
                sym_pi_board = np.rot90(pi_board, rot)
                sym_pi_board = np.rot90(sym_pi_board, rot, (2, 3))
                if flip:
                    sym_board = np.fliplr(sym_board)
                    sym_pi_board = np.fliplr(sym_pi_board)
                    sym_pi_board = np.flip(sym_pi_board, 2)
                sym_forms += [(sym_board, sym_pi_board.ravel())]
        return sym_forms
    
    def stringRepresentation(self, board):
        board = self.boardToClass(board)
        return board.toString()
    
    def stringRepresentationReadable(self, board):
        board = self.boardToClass(board)
        return board.toStringReadable()

    @staticmethod
    def display(board: Board): # TODO
        board_str = ""
        side_length = len(board)
        for row in range(side_length - 1, -1, -1):
            board_str += chr(ord('1') + row)
            board_str += " |"
            for col in range(side_length):
                board_str += chrs[board[row, col]] + "|"
            board_str += '\n'
            # board_str += "  |"
            # for col in range(side_length):
            #     board_str += "-+"
            # board_str += '\n'
        board_str += "  |"
        for col in range(side_length):
            board_str += chr(ord('a') + col) + "|"
        print(board_str)

# random play for testing

# class RandomPlayer():

#     def __init__(self, game):
#         self.game = game

#     def play(self, board, player):
#         idx = np.random.randint(0, self.game.getActionSize())
#         valid_moves = self.game.getValidMoves(board, player)
#         while valid_moves[idx] == 0:
#             idx = np.random.randint(0, self.game.getActionSize())
#         return idx

# while True:
#     game = AtaxxGame()
#     board = game.getInitBoard()
#     player = RandomPlayer(game)
#     player_idx = 1
#     print(game.stringRepresentationReadable(board))
#     while game.getGameEnded(board, player_idx) == 0:
#         board, player_idx = game.getNextState(board, player_idx, player.play(board, player_idx))
#         print(game.stringRepresentationReadable(board))