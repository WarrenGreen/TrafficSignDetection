import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

cafferoot = "path/to/caffe/root"
sys.path.insert(0, caffe2root + 'python')
import caffe


##
# Usage: python test_image.py pathToImage
#
# Forward feeds image through CNN and displays image with predicted street sign.
#
# Based largely on:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
##


# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

caffe.set_mode_cpu()

model_def = caffe2root + 'models/bvlc_alexnet/deploy.prototxt'
model_weights = cafferoot + 'models/bvlc_alexnet/caffe_alexnet_train_iter_1400.caffemodel'
labelsPath = "data/labels.txt"
imgPath = sys.argv[1]

labelsFile = open(labelsPath, 'r')
labels = {}
for line in labelsFile:
  label, num = line.split(" ")
  labels[num.strip()] = label

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('image_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(imgPath)
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

img = cv2.imread(imgPath)
cv2.putText(img,labels[str(output_prob.argmax())], (0,img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 5)
cv2.imshow("g",img)
cv2.waitKey(0)
print 'predicted class is:', output_prob.argmax()
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print output_prob[top_inds]