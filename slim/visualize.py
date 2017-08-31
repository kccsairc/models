from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

from io import BytesIO
from urllib2 import urlopen
import numpy as np
import sys, os

from PIL import Image
import matplotlib as mpl;mpl.use('Agg') # use Agg to avoid error (QXcbConnection: Could not connect to display)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from skimage.io import imsave

def get_image_data(url=None,img_path=None):
    raw_image = None
    if url != None:
        raw_image = urlopen(url).read()
    elif img_path != None:
        raw_image = tf.read_file(img_path)
    return tf.image.decode_jpeg(raw_image, channels=3)
    
slim = tf.contrib.slim

def createVisualization(checkpoint_path=None,model_name=None,num_classes=0,img_path=None):

  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()
    
    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
       model_name,
        num_classes=num_classes,
        is_training=False)
    
    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = network_fn.default_image_size
    sample_image = get_image_data(img_path=img_path)
    image = image_preprocessing_fn(sample_image, eval_image_size, eval_image_size)
    image  = tf.expand_dims(image, 0)

    logits,end_points = network_fn(image)
    predictions = tf.argmax(logits, 1)
    probabilities = tf.nn.softmax(logits)

    def vis_square(data, filename):
        """Take an array of shape (n, height, width) or (n, height, width, 3)
           and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
        # normalize data for display
        data = (data - data.min()) / (data.max() - data.min())
       
        print(len(data.shape)) 
        if len(data.shape) == 3:
            cmap = plt.get_cmap('jet')
            data = cmap(data)
    
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                   (0, 1), (0, 1))                 # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
        imsave(filename, data)


    #################################
    # Vizualize activation function #
    #################################
    def output_activation(session, end_points, layer_name, filename):
        import time
	start = time.time()

	# preprocess data
        data = session.run(end_points[layer_name])
	raw_data_shape = data.shape
	data = ((data - data.min()) / (data.max() - data.min())).T
	n_units, width, height, dim = data.shape

	# (n_units, width, height, 1) -> (n_units, width, height)
	if dim==1:
	    data = data.reshape([n_units, width, height])
	    cmap = plt.get_cmap('jet')
	    data = cmap(data)

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))
	data = np.pad(data, padding, mode='constant', constant_values=1)

	# tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

	data = np.transpose(data, (1, 0, 2))
	imsave(filename, data)
	elapsed_time = time.time() - start
        print("...save fig:", filename, elapsed_time, layer_name, data.shape, raw_data_shape)
     
    
    ####################
    # Print Probability #
    ####################
    sv = tf.train.Supervisor(logdir=checkpoint_path)
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with sv.managed_session(config=config) as sess:
        proba = sess.run(probabilities)[0]
        result = np.array([[i, p] for i, p in zip(range(num_classes), proba)])
        viz_directory = "visualized_layer/"
        layer_name_list=[]
        for key,val in end_points.iteritems():
            layer_name_list.append(key)
        print(layer_name_list)
        for l in layer_name_list:
            if 'Prediction' in l or 'prediction' in l or 'Logits' in l or 'logits' in l:
                continue
            else:
                layer_name = str(l).replace("/", "_")
                filename = viz_directory + layer_name + ".png"
                output_activation(sess, end_points, l, filename)
    return result

if __name__ == '__main__':
  tf.app.run()
