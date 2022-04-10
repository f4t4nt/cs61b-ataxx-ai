import random as rd
from signal import SIG_DFL
from . import AtaxxGame as Game

class RandomPlayer():

    def __init__(self, game):
        self.game = game

    def play(self, board):
        idx = rd.randint(0, self.game.getActionSize())
        valid_moves = self.game.getValidMoves(board, 1)
        while valid_moves[idx] == 0:
            idx = rd.randint(0, self.game.getActionSize())
        return idx

class HumanAtaxxPlayer():

    def __init__(self, game: Game):
        self.game = game
        self.SIDE_LENGTH = self.game.getBoardSize()[0]

    def play(self, board):
        self.game.display(board)
        valid = self.game.getValidMoves(board, 1)
        valid_move = {}
        for i in range(len(valid)):
            if valid[i]:
                move = self.game.idxToMove(i)
                col0 = chr(ord('1') + int(move[0]))
                row0 = chr(ord('a') + int(move[1]))
                col1 = chr(ord('1') + int(move[2]))
                row1 = chr(ord('a') + int(move[3]))
                valid_move[moveToStr((row0, col0, row1, col1))] = i
                valid_move[moveToStr((col0, row0, col1, row1))] = i
                # print("{}{}-{}{}".format(col0, row0, col1, row1))
        while True:
            move = input("Your move: ")
            if move == 'exit':
                return -1
            if move in valid_move:
                return valid_move[move]

            print("Invalid move")

def moveToStr(tupl):
    return tupl[1] + tupl[0] + '-' + tupl[3] + tupl[2]


class GreedyAtaxxPlayer(): # TODO

    def __init__(self, game):
        self.game = game

    def play(self, board):
        idx = rd.randint(0, self.game.getActionSize())
        valid_moves = self.game.getValidMoves(board, 1)
        while valid_moves[idx] == 0:
            idx = rd.randint(0, self.game.getActionSize())
        return idx