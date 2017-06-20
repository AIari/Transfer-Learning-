"""
    More info at https://www.tensorflow.org/tutorials/image_retraining
    Download inception model from 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    Place it in 'inception' folder
    'Bottleneck' is the layer before the final output layer which actually does the classification


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
# from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

model_dir = 'inception'
image_dir = 'train'
bottleneck_dir = 'bottlenecks'


def print_layers():
    for tensor in tf.get_default_graph().as_graph_def().node:
        print(tensor.name)


def create_bottleneck_file(sess, jpeg_data_tensor, bottleneck_tensor):
    file_list = []
    file_glob = os.path.join(image_dir, '*.jpg')
    file_list.extend(gfile.Glob(file_glob))

    for image in file_list:
        # getting the name of the training image
        image_name = str(image).split('/')[-1]
        path = bottleneck_dir + '/' + image_name + '.txt'
        print('Creating bottleneck at ' + path)

        image_data = gfile.FastGFile(image, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# Create graph from the inception V3 model
def create_inception_graph():
    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:  #extracting the  inception proto fine 
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())  #fill the tuple with the operations in the graph 
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='',
                                                                                             return_elements=[
                                                                                                 BOTTLENECK_TENSOR_NAME,
                                                                                                 JPEG_DATA_TENSOR_NAME,
                                                                                                 RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def main():
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

    sess = tf.Session()

    create_bottleneck_file(sess, jpeg_data_tensor, bottleneck_tensor)


if __name__ == '__main__':
    main()
