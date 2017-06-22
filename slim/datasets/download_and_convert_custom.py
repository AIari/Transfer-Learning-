#!/usr/bin/env python
r"""Converts custom data to TFRecords of TF-Example protos.

This module converts a custom data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

"""
The URL where the custom data can be downloaded. Update this with your dataset link.
Dataset should be as in the form of in this example dataset given under data/custom folder.
In custom folder make directories for each form of your classification and keep your data in the appropriate folder

"""
_DATA_URL = 'https://www.dropbox.com/s/g9x8tob8dekgf9y/xmas_photos.tar.gz?dl=0'

"""
The number of images in the validation set.
Update this with the size of your preferred validation set out of your dataset
"""
_NUM_VALIDATION = 20

# Seed for repeatability.
_RANDOM_SEED = 0

"""
The number of shards per dataset split.
Update this with the number of classes you have in your dataset
"""
_NUM_SHARDS = 2


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):  #this is clearly connected with the format how we save the data ser 
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    """
    Change 'custom_photos' with the name you need for the dataset
    """
    custom_root = os.path.join(dataset_dir, 'custom_photos')
    directories = []
    class_names = []
    for filename in os.listdir(custom_root):  #extracting the folders 
        path = os.path.join(custom_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    """
    :param dataset_dir:
    :param split_name:
    :param shard_id:
    :return:
    """
    """
    Change the custom with the name you need
    """
    output_filename = 'custom_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir): #We didn't change this 
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    """
    You'll not need to run this method if you're not using a remote dataset
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)
    """
    Change 'custom_photos' appropriately, should change according to the file you're downloading from the server
    """
    tmp_dir = os.path.join(dataset_dir, 'custom_photos')
    tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir):             
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):      
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    """
    Run this line if you've a remote dataset, otherwise keep it commented
    """
    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)  
   
    #it's important to save data set as relevent format. like folders and their 
    
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir) #class names and file names (extract photo files also)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))  #this is to change class name to IDs . Zip will asi
    #assign class names to numbers 

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_VALIDATION:]   #extract the training file names from the list 
    validation_filenames = photo_filenames[:_NUM_VALIDATION] #extract the val same as above 

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,  #converting the data set
                     dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    """
    Run this line if you've a remote dataset, otherwise keep it commented, this removes downloaded tar files
    """
    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the custom dataset!')
