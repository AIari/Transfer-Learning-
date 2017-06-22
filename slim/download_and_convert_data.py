r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=custom \
    --dataset_dir=/tmp/datasets/custom

```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_custom

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist", "custom"')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    # Add custom dataset_name
    if FLAGS.dataset_name == 'custom':
        download_and_convert_custom.run(FLAGS.dataset_dir)
    else:
        raise ValueError(
            'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)


if __name__ == '__main__':
    tf.app.run()
