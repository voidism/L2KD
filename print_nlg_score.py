from collections import OrderedDict
import sys
f = open(sys.argv[1]).readlines()
for line in f:
    if "score: " in line:
        line = line.strip().split("score: ")[-1]
        d = eval(line)
        print('\t'.join([str(x) for x in [d['e2enlg']['avg_rouge'], d['rnnlg.rest']['avg_rouge'], d['rnnlg.hotel']['avg_rouge'], d['rnnlg.tv']['avg_rouge'], d['rnnlg.laptop']['avg_rouge']]]))

