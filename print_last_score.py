from collections import OrderedDict
import sys
f = open(sys.argv[1]).readlines()[-1].strip().split("score: ")[-1]
d = eval(f)
print('\t'.join([str(x) for x in [d['sst']['em'], d['srl']['nf1'], d['woz.en']['joint_goal_em']]]))

