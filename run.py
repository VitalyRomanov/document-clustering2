rwww.win2.cn/g9
import argparse
import datetime
from analyzer.analyzer import Analyzer

parser = argparse.ArgumentParser(description='Utility for clustering textual documents')
parser.add_argument('-s', '--server', help='Run in server mode for article labeling',
                    action='store_true')
args = vars(parser.parse_args())


print("\n\n")
print(datetime.datetime.now().strftime('%d %b %G %I:%M%p'))
print("\n")

params = dict()
params['index_by'] = 'titles'
params['enable_filtering'] = False
params['resources'] = "./res/"
params['similarity_threshold'] = .4
params['server_mode'] = False


# parse argument
if args['server']:
    params['server_mode'] = True



def print_configuration(params):
    print("Configuration:")
    print("\tIndex by: %s" % params['index_by'])
    print("\tFiltering enabled: %s" % repr(params['enable_filtering']))
    print("\tResource folder: %s" % params['resources'])
    print("\tSimilarity threshold: %.2f" % params['similarity_threshold'])
    print("\tServer mode enabled: %s" % repr(params['server_mode']))


print_configuration(params)

a = Analyzer(params)
a.begin()