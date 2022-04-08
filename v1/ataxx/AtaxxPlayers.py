import random as rd
from signal import SIG_DFL

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

    def __init__(self, game):
        self.game = game
        self.SIDE_LENGTH = self.game.getBoardSize()[0]

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                move = self.game.idxToMove(i)
                col0 = chr(ord('1') + move[0])
                row0 = chr(ord('a') + move[1])
                col1 = chr(ord('1') + move[2])
                row1 = chr(ord('a') + move[3])
                print("{}{}-{}{}".format(col0, row0, col1, row1))
        while True:
            move = input("Your move: ")
            if move == 'exit':
                return -1
            move = move.split('-')
            if len(move) != 2:
                print("Invalid move")
                continue
            col0 = ord(move[0][0]) - ord('1')
            row0 = ord(move[0][1]) - ord('a')
            col1 = ord(move[1][0]) - ord('1')
            row1 = ord(move[1][1]) - ord('a')
            if not (0 <= col0 < self.SIDE_LENGTH and \
                0 <= row0 < self.SIDE_LENGTH and \
                0 <= col1 < self.SIDE_LENGTH and \
                0 <= row1 < self.SIDE_LENGTH):
                print("Invalid move")
                continue
            move = self.game.moveToIdx((col0, row0, col1, row1))
            if valid[move]:
                return move
            print("Invalid move")

class GreedyAtaxxPlayer(): # TODO

    def __init__(self, game):
        self.game = game

    def play(self, board):
        idx = rd.randint(0, self.game.getActionSize())
        valid_moves = self.game.getValidMoves(board, 1)
        while valid_moves[idx] == 0:
            idx = rd.randint(0, self.game.getActionSize())
        return idx