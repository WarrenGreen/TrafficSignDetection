#CNN Traffic Sign Recognition <br/>

Using the caffe framework to train a CNN to detect US traffic street signs. Using the alexnet design, 98.5% accuracy can be reached within several hundred iterations. While all of the CNN data is included to feed a test image forward, instructions to train the net yourself is included.

First the caffe root path will have to be filled into several of the python files.

To run a test image:

  python test_image.py pathToImage

<br/>

To train your own net:
<code>
  # Download and unzip LISA dataset
  wget http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip
  unzip signDatabasePublicFramesOnly.zip

  # Prepare LISA dataset to be ingested into leveldbs
  python dataprep.py ./signDatabasePublicFramesOnly/ ./data

  # Build leveldbs
  # If there are currently leveldb folders, this will fail.
  # DATA param in create_imagenet.sh must first be changed to point to data directory you just created
  $caffe_root/examples/imagenet/create_imagenet.sh

  # Compute image mean 
  # DATA param in make_imagenet_mean.sh must first be changed to point to data directory you just created
  $caffe_root/examples/imagenet/make_imagenet_mean.sh

  # Start training from scratch
  caffe train --solver=alexnet/solver.prototxt

  # Pickup training from previous snapshot
  caffe train --solver=alexnet/solver.prototxt  --snapshot=caffe_alexnet_train_iter_1500.solverstate

  # Create index between CNN output value and label
  python labelIndexCreation.py ./data

  # Convert image mean from binaryproto to npy
  python convert.py imagenet_mean.binaryproto image_mean.npy

  # Classify and display image
  python test_image.py pathToImage
</code>

<br/>

# Image Results

![Stop](/results/Screen Shot 2016-12-28 at 11.45.13 AM.png?raw=true "")
![Left Turn](/results/Screen Shot 2016-12-28 at 11.46.36 AM.png?raw=true "")
![Pedestrian Crossing](/results/Screen Shot 2016-12-28 at 11.47.12 AM.png?raw=true "")
![Merge](/results/Screen Shot 2016-12-28 at 11.48.06 AM.png?raw=true "")
![Speed Limit 35](/results/Screen Shot 2016-12-28 at 11.49.04 AM.png?raw=true "")
![Speed Limit 45](/results/Screen Shot 2016-12-28 at 11.49.46 AM.png?raw=true "")


