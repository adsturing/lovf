#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(12345)
np.random.seed(12345)
random.seed(12345)
import json
import re
import collections
import six

from sklearn.metrics import roc_auc_score


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def build_embeddings(emb_name,
                     vocab_size,
                     emb_dim,
                     zero_padding=True,
                     initializer=create_initializer(0.02)):
  embeddings = tf.get_variable(
    name=emb_name,
    shape=[vocab_size, emb_dim],
    initializer=initializer
  )
  if zero_padding:
    zero_emb = tf.zeros((1, emb_dim), tf.float32)
    embeddings = tf.concat([zero_emb, embeddings], axis=0)
  return embeddings


def dcg_score(y_true, y_score, k=10, gains="exponential"):
  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k])

  if gains == "exponential":
    gains = 2 ** y_true - 1
  elif gains == "linear":
    gains = y_true
  else:
    raise ValueError("Invalid gains option.")

  discounts = np.log2(np.arange(len(y_true)) + 2)
  return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
  best = dcg_score(y_true, y_true, k, gains)
  actual = dcg_score(y_true, y_score, k, gains)
  return actual / (best + 1e-8)

def compute_metrics(labels_preds, top_k_list):
  labels, preds, loss = zip(*labels_preds)
  labels = list(labels)
  metrics = {}
  num_pos = sum(labels)
  if num_pos == len(labels) or num_pos == 0:
    return None
  else:
    metrics['gauc'] = roc_auc_score(labels, preds)
  for top_k in top_k_list:
    ndcg_k = ndcg_score(labels, preds, k=top_k)
    metrics['ndcg@{}'.format(top_k)] = ndcg_k

  return metrics

def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == 'lrelu':
    return tf.nn.leaky_relu
  elif act == "tanh":
    return tf.tanh
  elif act == "sigmoid":
    return tf.nn.sigmoid
  elif act =='elu':
    return tf.nn.elu
  else:
    raise ValueError("Unsupported activation: %s" % act)


def mlp(input, params,first_bn = True,layer_list=None,op_name = ""):
  """
  Implementation of multi-layer perceptron
  :param input: (Tensor) input of mlp
  :param mlp_layers: (str) layers used in mlp, e.g., '256,128'
  :param dropout_prob: (float) ratio of network to drop
  :param activation: (str) activation function to use
  :return: output of mlp
  """
  act_func = get_activation(params.agg_act)
  if layer_list != None:
    mlp_layers = layer_list
  else:
    mlp_layers = [int(unit) for unit in params.agg_layers.split(',')]
  mlp_net = input
  if params.use_bn:
    mlp_net = tf.layers.batch_normalization(
      mlp_net, training=params.is_training)
    if act_func is not None:
      mlp_net = act_func(mlp_net, name=op_name+'act_0')


  for i in range(len(mlp_layers)):
    mlp_net = tf.layers.dense(mlp_net, mlp_layers[i], name=op_name+'fc_{}'.format(i))
    mlp_net = tf.nn.dropout(mlp_net, 1.0 - params.dropout_prob)
    if params.use_bn:
      mlp_net = tf.layers.batch_normalization(
        mlp_net, training=params.is_training)
    if act_func is not None:
      mlp_net = act_func(mlp_net, name=op_name+'act_{}'.format(i+1))
  return mlp_net





