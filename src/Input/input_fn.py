# _*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle
import copy
import random
random.seed(12345)
import json

import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf

tf.set_random_seed(12345)
np.random.seed(12345)
from collections import defaultdict

import sys
sys.path.append('./src')
from Utils import utils
from Utils.params_config import *

class BaseData(object):
  def __init__(self, params, filename):
    self.params = params
    output_types = {key: tf.int32 for key in params.feature_name.split(',')}
    self.output_types = output_types
    output_types.update({'global_group_id': tf.int32})
    self.group_data = self.load_dataset_from_disk(filename)
    self.data_size = len(self.group_data)
    self.indexes = list(range(self.data_size))
    print('data size is : {}'.format(self.data_size))

  def get_data(self, mode):
    logging.info('creating {} tf.data.dataset instance...'.format(mode))

    dataset = tf.data.Dataset.from_generator(
      lambda: self.data_generator(),
      output_types=self.output_types
    )

    dataset = dataset.batch(1).prefetch(self.params.buffer_size)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    features = iterator.get_next()
    features = {key: value[0] for key, value in features.items()}
    features['iterator_init_op'] = init_op
    return features

  def shuffle_dataset(self):
    random.shuffle(self.indexes)

  def load_dataset_from_disk(self, filename):
    logging.info('loading {} from disk...'.format(filename))
    path = os.path.join(self.params.data_dir, filename)
    with open(path, 'rb') as reader:
      group_data = pickle.load(reader)
    return group_data

  def data_generator(self):
    batch_size = self.params.batch_size
    feature_name = self.params.feature_name.split(',')
    global_group_id = 0
    for i in range(0, self.data_size, batch_size):
      batch_data = defaultdict(list)
      start = i
      end = self.data_size if i + batch_size > self.data_size else i + batch_size
      group_id, group_start = 0, 1
      for j in range(start, end):
        line = self.group_data[self.indexes[j]]
        user, query = line[0], line[1]
        ori_group_size = len(line[2])  #label的数量
        if ori_group_size > self.params.max_group_len:
            group_size = self.params.max_group_len
        else:
            group_size = ori_group_size
        user_query = [[user for _ in range(group_size)], [query for _ in range(group_size)]]
        
        for k, key in enumerate(feature_name):
          if k < 2:
            batch_data[key].extend(user_query[k]) #将user_query信息放置于batch_data
          elif k < 4:
            label_info = line[k]
            batch_data[key].extend(label_info[:group_size])
          else:
            break
        expo_info = line[4:4+group_size]  #所有展现ad的特征
        for k in range(group_size):
          for key, value in zip(feature_name[4:], expo_info[k]):
            batch_data[key].append(value)
        global_group_ids = [global_group_id for _ in range(group_size)]
        batch_data['global_group_id'].extend(global_group_ids)
        global_group_id += 1
      yield batch_data







