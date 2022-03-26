from __future__ import print_function
import sys
sys.path.append('..')
from game import Game
from .AtaxxLogic import Board
import numpy as np

class AtaxxGame(Game):
    
    def __init__(self, side_length = 7, jump_limit = 25, wall_p = 0.2):
        self.SIDE_LENGTH = side_length
        self.jump_limit = jump_limit
        self.wall_p = wall_p

    def getInitBoard(self):
        self.init_board = Board(self.SIDE_LENGTH, self.jump_limit, self.wall_p)
        return self.init_board.board

    def getBoardShape(self):
        self.board_shape = (self.SIDE_LENGTH, self.SIDE_LENGTH)
        return self.board_shape

    def getActionSize(self):
        self.action_size = 25 * self.SIDE_LENGTH ** 2
        return self.action_size

    def idxToMove(self, move):
        col0 = move // (self.SIDE_LENGTH * 25) + 2
        row0 = (move % (self.SIDE_LENGTH * 25)) // 25 + 2
        dc = (move % 25) // 5 - 2
        dr = move % 5 - 2
        if dc == 0 and dr == 0:
            col0, row0, col1, row1 = [0] * 4
        else:
            col1, row1 = col0 + dc, row0 + dr
        return (col0, row0, col1, row1)

    def boardToClass(self, board):
        return Board(board)

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
        return board.toString()
    
    def stringRepresentationReadable(self, board):
        return board.toStringReadable()

    @staticmethod
    def display(board): # TODO
        pass