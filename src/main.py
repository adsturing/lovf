#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from Utils.params_config import *
from Input.input_fn import *
from Model.model_fn import *
from Solver.solver import train_and_evaluate 
from Utils import utils

import argparse
import logging
import os

tf.set_random_seed(12345)
np.random.seed(12345)

def process_args():
  # load hyper-parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_dir', type=str,
                      default='.')
  parser.add_argument('--model_name', type=str, default='ours')
  args = parser.parse_args()
  return args


def get_num_info():
  num_info = {
    'num_skus': 10153389,
    'num_venders': 20693,
    'num_cates': 4374,
    'num_brands': 151870,
    'num_queries': 2779127,
    'num_users': 9174130,
    'num_prices': 462
  }
  return num_info


def main():
  args = process_args()
  params = BaseParams(args)

  num_info = get_num_info()
  params.update(num_info)
  params.make_model_dir()

  # initialize dataset and model
  logging.info('creating datasets....')
  input_fn_train = BaseData(params, params.train_filename)
  input_fn_eval = BaseData(params, params.test_filename)
  model_fn = lovf_Model(params)
  train_and_evaluate(input_fn_train, input_fn_eval, model_fn, params)

if __name__ == '__main__':
    main()
