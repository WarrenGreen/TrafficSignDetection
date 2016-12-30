import sys
import numpy as np

caffe_root="caffe/root/dir"
sys.path.insert(0, caffe_root + 'python')
import caffe

##
# Usage: python convert_protomean.py proto.mean out.npy
#
# Convert binaryproto mean image file to npy for CNN forward usage.
#
# Script taken from mafiosso from the Caffe issue tracker link below.
#   https://github.com/BVLC/caffe/issues/290
#
##


if len(sys.argv) != 3:
  print "Usage: python convert.py proto.mean out.npy"
  sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )