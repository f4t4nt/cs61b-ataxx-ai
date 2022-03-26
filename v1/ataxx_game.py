from __future__ import print_function
from lib2to3.pgen2.token import AT
# import sys
# sys.path.append('..')
from ataxx_logic import AtaxxBoard
import numpy as np

class AtaxxGame:
    
    def __init__(self, side_length = 7, jump_limit = 25, wall_p = 0.2):
        self.SIDE_LENGTH = side_length
        self.jump_limit = jump_limit
        self.wall_p = wall_p

    def get_init_board(self):
        self.init_board = AtaxxBoard(self.SIDE_LENGTH, self.jump_limit, self.wall_p)
        return self.init_board.board

    def get_board_shape(self):
        self.board_shape = (self.SIDE_LENGTH, self.SIDE_LENGTH)
        return self.board_shape

    def get_action_size(self):
        self.action_size = 25 * self.SIDE_LENGTH ** 2
        return self.action_size

    def idx_to_move(self, move):
        col0 = move // (self.SIDE_LENGTH * 25) + 2
        row0 = (move % (self.SIDE_LENGTH * 25)) // 25 + 2
        dc = (move % 25) // 5 - 2
        dr = move % 5 - 2
        if dc == 0 and dr == 0:
            col0, row0, col1, row1 = [0] * 4
        else:
            col1, row1 = col0 + dc, row0 + dr
        return (col0, row0, col1, row1)

    def board_to_class(self, board):
        return AtaxxBoard(board)

    def get_next_state(self, board, player, move):
        board = self.board_to_class(board)
        col0, row0, col1, row1 = self.idx_to_move(move)
        board.make_move(player, (col0, row0, col1, row1))
        return (board.board, -player)
    
    def get_valid_moves(self, board, player):
        board = self.board_to_class(board)
        valid_moves = np.zeros(25 * self.SIDE_LENGTH ** 2)
        valid_moves_list = board.get_moves(player)
        for move in range(25 * self.SIDE_LENGTH ** 2):
            col0, row0, col1, row1 = self.idx_to_move(move)
            if (col0, row0, col1, row1) in valid_moves_list:
                valid_moves[move] = 1
        return valid_moves
    
    def get_game_ended(self, board, player):
        board = self.board_to_class(board)
        winner, ended = board.get_winner()
        if ended:
            if winner == player:
                return 1
            elif winner == -player:
                return -1
            elif winner == 0:
                return -0.1
        else:
            return 0

    def get_canonical_form(self, board, player):
        board = self.board_to_class(board)
        return board.pieces * player + board.walls

    def get_symmetries(self, board, pi):
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
    
    def string_representation(self, board):
        return board.to_string()

board = AtaxxBoard()
game = AtaxxGame()
game.get_symmetries(board, game.get_valid_moves(board, 1))
x = 1