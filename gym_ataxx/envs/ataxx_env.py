import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple
import random

class AtaxxEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    SIDE_LENGTH, JUMP_LIMIT = 7, 25
    SIDE_LENGTH_EXTENDED = SIDE_LENGTH + 4

    def index(self, col, row):
        return (col + 2) * self.SIDE_LENGTH_EXTENDED + (row + 2)

    def neighbor(self, sq, dc, dr):
        return sq + dr + dc * self.SIDE_LENGTH_EXTENDED

    # EMPTY: -1
    # WHITE:  0
    # BLACK:  1
    # WALL:   2
    def getSquareSq(self, sq):
        return self.board[sq]

    def getSquareCR(self, col, row):
        return self.getSquareSq(self.index(col, row))

    def setSquareSq(self, sq, ci):
        self.board[sq] = ci

    def setSquareCR(self, col, row, ci):
        self.setSquareSq(self.index(col, row), ci)

    def getPieces(self, ci):
        return self.pieces[ci]

    def whitePieces(self):
        return self.getPieces(0)

    def blackPieces(self):
        return self.getPieces(1)

    def incrPieces(self, ci, k):
        self.pieces[ci] += k

    def incrWhite(self, k):
        self.incrPieces(0, k)

    def incrBlack(self, k):
        self.incrPieces(1, k)

    # GAME ALIVE: -1
    # WHITE WINS:  0
    # BLACK WINS:  1
    # DRAW:        2
    def getWinner(self):
        if self.numJumps == self.JUMP_LIMIT:
            self.winner = 2
        elif self.whitePieces() == 0:
            self.winner = 1
        elif self.blackPieces() == 0:
            self.winner = 0
        elif self.canMove(0) or self.canMove(1):
            self.winner = -1
        elif self.whitePieces() > self.blackPieces():
            self.winner = 0
        elif self.whitePieces() < self.blackPieces():
            self.winner = 1
        elif self.whitePieces() == self.blackPieces():
            self.winner = 2
        return self.winner

    # c0: m[0]
    # r0: m[1]
    # c1: m[2]
    # r1: m[3]
    # c0 = -1 IFF M IS A PASS
    def getMoves(self, ci):
        moves = [ ]
        for col in range(self.SIDE_LENGTH):
            for row in range(self.SIDE_LENGTH):
                sq = self.index(col, row)
                if self.getSquareSq(sq) == ci:
                    for dc in range(-2, 3):
                        for dr in range(-2, 3):
                            if self.getSquareSq(self.neighbor(sq, dc, dr)) == -1:
                                moves += [ [ col, row, col + dc, row + dr  ] ]
        return moves

    def getWhiteMoves(self):
        self.whiteMoves = self.getMoves(0)
        return self.whiteMoves

    def getBlackMoves(self):
        self.blackMoves = self.getMoves(1)
        return self.blackMoves

    def canMove(self, ci):
        return len(self.getMoves(ci)) > 0

    def canWhiteMove(self):
        return self.canMove(0)

    def canBlackMove(self):
        return self.canMove(1)

    def isExtend(self, m):
        return max(abs(m[0] - m[2]), abs(m[1] - m[3])) == 1

    def isJump(self, m):
        return max(abs(m[0] - m[2]), abs(m[1] - m[3])) == 2

    def legalMove(self, ci, m):
        if ci != self.turn:
            raise
            return False
        if m[0] == -1:
            return not self.canMove(ci)
        else:
            return self.getSquareSq(self.index(m[0], m[1])) == ci and \
                self.getSquareSq(self.index(m[2], m[3])) == -1 and \
                (self.isExtend(m) or \
                self.isJump(m))

    def makeMove(self, ci, m):
        if not self.legalMove(ci, m):
            raise
            return False
        if m[0] != -1:
            if self.isJump(m):
                self.numJumps += 1
                self.setSquareCR(m[0], m[1], -1)
                self.incrPieces(ci, -1)
            else:
                self.numJumps = 0
            sq = self.index(m[2], m[3])
            self.setSquareSq(sq, ci)
            self.incrPieces(ci, 1)
            for dc in range(-1, 2):
                for dr in range(-1, 2):
                    if self.getSquareSq(self.neighbor(sq, dc, dr)) == 1 - ci:
                        self.setSquareSq(self.neighbor(sq, dc, dr), ci)
                        self.incrPieces(ci, 1)
                        self.incrPieces(1 - ci, -1)
        self.turn = 1 - self.turn
        return True

    def __init__(self):
        self.action_space = Tuple((Discrete(self.SIDE_LENGTH), Discrete(self.SIDE_LENGTH), Discrete(5, start = -2), Discrete(5, start = -2)))
        self.observation_space = MultiDiscrete([4] * self.SIDE_LENGTH ** 2)
        
    def getObservation(self):
        observation = [ ]
        for row in range(self.SIDE_LENGTH):
            for col in range(self.SIDE_LENGTH):
                observation += [ self.getSquareCR(col, row) ]
        return observation
        
    def step(self, action):
        if action[2] == 0 and action[3] == 0:
            m = [ -1, -1, -1, -1 ]
        else:
            m = [ action[0], action[1], action[0] + action[2], action[1] + action[3] ]
        if not self.legalMove(0, m):
            return (self.getObservation(), -1000, False, { "winner": "n/a" } )
        else:
            self.makeMove(0, m)
            if self.canBlackMove():
                m = random.choice(self.getBlackMoves())
            else:
                m = [ -1, -1, -1, -1 ]
            self.makeMove(1, m)
            reward = -1
            done = (self.getWinner() != -1)
            winner = "n/a"
            if self.winner == 0:
                reward = 100
                winner = "white"
            elif self.winner == 1:
                reward = -100
                winner = "black"
            elif self.winner == 2:
                reward = -50
                winner = "draw"
            return ( self.getObservation(), reward, done, { "winner": winner } )

    def reset(self):
        self.board = [ 2 ] * self.SIDE_LENGTH_EXTENDED ** 2
        for col in range(self.SIDE_LENGTH):
            for row in range(self.SIDE_LENGTH):
                self.setSquareCR(col, row, -1)
        self.setSquareCR(0, 0, 1)
        self.setSquareCR(0, self.SIDE_LENGTH - 1, 0)
        self.setSquareCR(self.SIDE_LENGTH - 1, 0, 0)
        self.setSquareCR(self.SIDE_LENGTH - 1, self.SIDE_LENGTH - 1, 1)
        self.turn = 0
        self.pieces = [ 2, 2 ]
        self.numJumps = 0
        self.winner = -1
        self.whiteMoves = [ ]
        self.blackMoves = [ ]
        return self.getObservation()

    def render(self, mode='human', close=False):
        board = ""
        for row in range(self.SIDE_LENGTH - 1, -1, -1):
            board += str(row + 1)
            for col in range(self.SIDE_LENGTH):
                board += ' '
                if self.getSquareCR(col, row) == -1:
                    board += '-'
                elif self.getSquareCR(col, row) == 0:
                    board += '○'
                elif self.getSquareCR(col, row) == 1:
                    board += '●'
                elif self.getSquareCR(col, row) == 2:
                    board += '■'
            board += '\n'
        board += ' '
        for col in range(self.SIDE_LENGTH):
            board += ' '
            board += chr(ord('a') + col)
        board += '\n'
        board += "○ "
        board += str(self.whitePieces())
        for i in range(len(str(self.whitePieces())) + 2, \
            2 * self.SIDE_LENGTH - len(str(self.blackPieces())) - 1):
            board += ' '
        board += str(self.blackPieces())
        board += " ●"
        board += '\n'
        print(board)
        return board