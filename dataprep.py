import os
from shutil import copyfile
import sys

##
# Usage: python dataprep.py inputPath destinationPath
#
# Prepare data from LISA sign dataset to be ingested to the CNN.
# Groups data into label based directory system and creates train.txt and val.txt indices.
#
# e.g.
#   /data/
#     -> leftTurn/
#           -> img1.png
#           -> img2.png
#     -> rightTurn/
#           -> img1.png
#           -> img2.png
#           -> ...
#     -> ...
##

INPUT_PATH = sys.argv[1]
DEST_PATH = sys.argv[2]

TRAIN = "train"
VAL = "val"

valuation_ratio = 0.2 # must ratio > .10
valuation_const = valuation_ratio * 10

trainIndex = open(os.path.join(DEST_PATH, "train.txt"), 'w')
valIndex = open(os.path.join(DEST_PATH, "val.txt"), 'w')

labels = {}

def countInc(count):
  count += 1
  if(count >= 10):
    count = 0

  return count

count = 0
for subdir, dirs, files in os.walk(INPUT_PATH):
  for file in files:
    if file.endswith(".png"):
      label = "na"
      if(file.startswith("nosign")):
        label = "nosign"
      else:
        label = file.split("_")[0]

      count = countInc(count)

      if label not in labels:
        labels[label] = len(labels)

      indexLabel = labels[label]

      if count < valuation_const:
        outputdir = os.path.join(DEST_PATH, VAL, label)
        valIndex.write(os.path.join(label, file)+" "+str(indexLabel)+"\n")
      else:
        outputdir = os.path.join(DEST_PATH, TRAIN, label)
        trainIndex.write(os.path.join(label, file)+" "+str(indexLabel)+"\n")

      if not os.path.exists(outputdir):
        os.makedirs(outputdir)
      copyfile(os.path.join(subdir, file), os.path.join(outputdir, file))

trainIndex.close()
valIndex.close()