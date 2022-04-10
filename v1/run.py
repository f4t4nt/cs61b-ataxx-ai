import main
import ataxx

main.args['load_folder_file'] = ('./temp', 'best.pth (5).tar')
main.args['against_model'] = ('./temp', 'best.pth (2).tar')
main.args['checkpoint'] = './temp/'
main.args['firstSelfPlayNumEps'] = 30
main.args['maxDepth'] = 40
main.args['maxTurns'] = 60
main.args['arenaCompare'] = 30
main.args['numItersForTrainExamplesHistory'] = 9
main.args['load_model'] = True
main.args['play_game'] = True
ataxx.pytorch.NNet.args['epochs'] = 20

main.main()