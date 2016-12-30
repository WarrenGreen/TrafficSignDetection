import os
import sys

##
# Usage: python labelIndexCreation.py pathToVal.txt
#
# Creates a label index file to correspond labels to their numerical values
# to be used with CNN forward valuations.
#
# e.g.
#   leftTurn 3
#   yield 4
#   rightTurn 5
#   stop 6
##

INPUT_PATH = sys.argv[1]

valIndex = open(os.path.join(INPUT_PATH, "val.txt"), 'r')
labelOut = open(os.path.join(INPUT_PATH, "labels.txt"), 'w')

imgLabels = {}

for line in valIndex:
  label = line.split("/")[0]
  if label not in imgLabels:
    imgLabels[label] = line.split(" ")[1]

for key in imgLabels:
  labelOut.write(key+" "+imgLabels[key])

labelOut.close()