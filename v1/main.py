import logging

import coloredlogs

import numpy as np

from Arena import Arena
from MCTS import MCTS

from Coach import Coach
from ataxx.AtaxxGame import AtaxxGame as Game
from ataxx.AtaxxPlayers import HumanAtaxxPlayer as Player
from ataxx.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'firstSelfPlayNumEps': 500, # Number of complete self-play games to simulate during a new iteration.
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'maxDepth': 50,
    'maxTurns': 100,

    'checkpoint': './temp/',
    'load_model': True,
    'against_model': None,
    'load_folder_file': ('temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'play_game': True
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')

    if args.play_game:
        if args.against_model is None:
            pmcts0 = MCTS(g, nnet, args)
            hplayer = Player(g)
            arena = Arena(
                lambda x: np.argmax(pmcts0.getActionProb(x, temp=0)),
                lambda x: hplayer.play(x),
                g)
            win_state = arena.playGame(args.maxTurns)
            print(win_state)
            arena = Arena(
                lambda x: hplayer.play(x),
                lambda x: np.argmax(pmcts0.getActionProb(x, temp=0)),
                g)
            win_state = arena.playGame(args.maxTurns)
            print(win_state)
        else:
            pmcts0 = MCTS(g, nnet, args)
            nnet1 = nn(g)
            nnet1.load_checkpoint(args.against_model[0], args.against_model[1])
            pmcts1 = MCTS(g, nnet1, args)
            arena = Arena(
                lambda x: np.argmax(pmcts0.getActionProb(x, temp=0)),
                lambda x: np.argmax(pmcts1.getActionProb(x, temp=0)),
                g,
                g.display)
            win_state = arena.playGame(args.maxTurns, verbose=True)
            print(win_state)
            arena = Arena(
                lambda x: np.argmax(pmcts1.getActionProb(x, temp=0)),
                lambda x: np.argmax(pmcts0.getActionProb(x, temp=0)),
                g,
                g.display)
            win_state = arena.playGame(args.maxTurns, verbose=True)
            print(win_state)
    else:
        c = Coach(g, nnet, args)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process 🎉')
        c.learn()


if __name__ == "__main__":
    main()
