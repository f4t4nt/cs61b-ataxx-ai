import random as rd

class RandomPlayer():

    def __init__(self, game):
        self.game = game

    def play(self, board):
        idx = rd.randint(0, self.game.get_action_size())
        valid_moves = self.game.get_valid_moves(board, 1)
        while valid_moves[idx] == 0:
            idx = rd.randint(0, self.game.get_action_size())
        return idx

class HumanAtaxxPlayer(): # TODO

    def __init__(self, game):
        self.game = game

    def play(self, board):
        idx = rd.randint(0, self.game.get_action_size())
        valid_moves = self.game.get_valid_moves(board, 1)
        while valid_moves[idx] == 0:
            idx = rd.randint(0, self.game.get_action_size())
        return idx

class GreedyAtaxxPlayer(): # TODO

    def __init__(self, game):
        self.game = game

    def play(self, board):
        idx = rd.randint(0, self.game.get_action_size())
        valid_moves = self.game.get_valid_moves(board, 1)
        while valid_moves[idx] == 0:
            idx = rd.randint(0, self.game.get_action_size())
        return idx