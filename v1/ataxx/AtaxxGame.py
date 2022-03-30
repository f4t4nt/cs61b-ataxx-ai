from __future__ import print_function
import sys

from scipy import rand
sys.path.append('..')
from Game import Game
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

    def getBoardSize(self):
        self.board_shape = (self.SIDE_LENGTH, self.SIDE_LENGTH)
        return self.board_shape

    def getActionSize(self):
        self.action_size = 25 * self.SIDE_LENGTH ** 2
        self.idx_denom = 25 * self.SIDE_LENGTH
        return self.action_size

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
        moves1_count = 0
        moves2_count = 0
        valid_moves1 = np.zeros(25 * self.SIDE_LENGTH ** 2)
        valid_moves2 = np.zeros(25 * self.SIDE_LENGTH ** 2)
        valid_moves_list = board.getMoves(player)
        for move in range(25 * self.SIDE_LENGTH ** 2):
            col0, row0, col1, row1 = self.idxToMove(move)
            if (col0, row0, col1, row1) in valid_moves_list:
                if col1 == col0 and row1 == row0 and (col0 or row0):
                    pass
                if abs(col1-col0) == 2 or abs(row1 - row0) == 2:
                    valid_moves2[move] = 1
                    moves2_count += 1
                else:
                    valid_moves1[move] = 1
                    moves1_count += 1
 
        if moves1_count == 0:
            return valid_moves2
        if moves2_count == 0:
            return valid_moves1
        
        ratio = 1
        if moves1_count > 5:
            ratio = moves1_count / (2 * moves2_count)
        else:
            ratio = moves1_count / moves2_count

        ratio = min(ratio, 1.)

        valid_moves2 = valid_moves2 * ratio
        valid_moves1 += valid_moves2
        return valid_moves1
    
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
    def display(board): # TODO
        pass

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