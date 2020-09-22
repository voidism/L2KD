from collections import OrderedDict
import sys
f = open(sys.argv[1]).readlines()
for line in f:
    if "score: " in line:
        line = line.strip().split("score: ")[-1]
        d = eval(line)
        print('\t'.join([str(x) for x in [d['woz.en']['joint_goal_em'], d['cnn_dailymail']['avg_rouge'], d['wikisql']['lfem']]]))

