#_*_coding:utf-8_*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
tf.set_random_seed(12345)
np.random.seed(12345)

from Utils import utils
import logging
import copy

class BaseModel(object):
  def __init__(self, params):
    self.params = params
    global_step = tf.train.get_or_create_global_step()
    self.learning_rate = tf.train.piecewise_constant(global_step, [8200], [0.001, 0.001])
    self.lamda = tf.train.piecewise_constant(global_step, [2000, 4000, 6000, 8000], [0.4, 0.8, 1.2, 1.5, 1.6])
    self.beta = tf.train.piecewise_constant(global_step, [2000, 4000, 6000, 8000], [1.6, 1.5, 1.2, 0.8, 0.4])

  def build_embedding_layer(self, params):
    self.sid_embeddings = utils.build_embeddings(
      emb_name='sku_id_embeddings',
      vocab_size=params.num_skus,
      emb_dim=params.emb_dim
    )
    self.cid_embeddings = utils.build_embeddings(
      emb_name='cate_id_embeddings',
      vocab_size=params.num_cates,
      emb_dim=params.emb_dim
    )
    self.vid_embeddings = utils.build_embeddings(
      emb_name='vender_id_embeddings',
      vocab_size=params.num_venders,
      emb_dim=params.emb_dim
    )
    self.bid_embeddings = utils.build_embeddings(
      emb_name='brand_id_embeddings',
      vocab_size=params.num_brands,
      emb_dim=params.emb_dim
    )
    self.qid_embeddings = utils.build_embeddings(
      emb_name='query_embeddings',
      vocab_size=params.num_queries,
      emb_dim=params.emb_dim
    )
    self.uid_embeddings = utils.build_embeddings(
      emb_name='user_embeddings',
      vocab_size=params.num_users,
      emb_dim=params.emb_dim
    )
    self.price_embeddings = utils.build_embeddings(
      emb_name='price_embeddings',
      vocab_size=params.num_prices,
      emb_dim=params.emb_dim
    )


  def build_train_operation(self, params, train_vars=None):
    for var in tf.trainable_variables():
      logging.info('{}, {}'.format(var.name, var.shape))
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-05, use_locking=True)

    tvars = tf.trainable_variables()
    grads = tf.gradients(self.loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
    self.train_op = train_op


    
# For a fair comparison, we set the embedding in advertising and organic scenarios to half the original size through hash conflicts to ensure that the total size of the embedding with other methods remains unchanged.  
class lovf_Model(BaseModel):
  def __init__(self, params):
    super(lovf_Model, self).__init__(params)


  def __call__(self, inputs, is_training, reuse=False):
    params = copy.deepcopy(self.params)
    mode = 'train' if is_training else 'valid'
    self.is_training =  is_training
    params.is_training = is_training
    params.dropout_prob = 0.0
    tf.summary.scalar('dropout_prob', params.dropout_prob)
    self.labels_all = tf.cast(inputs['label'], tf.float32)
    
    self.input_type = tf.cast(inputs['input_type'], tf.float32)
    
    self.input_type = tf.reshape(self.input_type,[-1,1])


    self.mask_main = tf.squeeze(tf.greater(self.input_type,0),1)
    self.mask_ad = tf.squeeze(tf.less(self.input_type,1),1)

    self.labels = tf.boolean_mask(self.labels_all,self.mask_ad)
    self.main_labels = tf.boolean_mask(self.labels_all,self.mask_main)



    with tf.variable_scope('Rank', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('embedding_layer'):
        self.build_embedding_layer(params)

      with tf.variable_scope('input_layer'):
        self.build_input_layer(inputs, params)

      with tf.variable_scope('ranking_layer'):
        self.build_ranking_layer(params)

      with tf.variable_scope('loss_layer'):
        self.build_loss_layer(params)

      inputs["global_group_id"] = tf.boolean_mask(inputs["global_group_id"],self.mask_ad)

      model_spec = inputs
      model_spec["label"] = self.labels
      model_spec['preds'] = self.preds
      model_spec['loss'] = self.loss
      model_spec['log_loss'] = self.log_loss

      self.build_train_operation(params)
      model_spec['train_op'] = self.train_op
      
      model_spec['learning_rate'] = self.learning_rate 
      
      return model_spec

  def build_ranking_layer(self, params):
    main_mlp_input = tf.concat(
      [self.main_user_emb, self.main_item_emb, self.main_query_emb], axis=1)

    ad_mlp_input = tf.concat(
      [self.ad_user_emb, self.ad_item_emb, self.ad_query_emb,self.main_user_emb, self.main_item_emb, self.main_query_emb], axis=1)
    ad_mlp_input = tf.boolean_mask(ad_mlp_input, self.mask_ad)

    main_mlp_output = utils.mlp(input=main_mlp_input, params=params,layer_list = [256],op_name = "main_1")
    main_bias_1 = tf.boolean_mask(main_mlp_output, self.mask_ad) 
    main_mlp_output = utils.mlp(input=main_mlp_output,params=params,layer_list = [128],op_name = "main_2")
    main_bias_2 = tf.boolean_mask(main_mlp_output, self.mask_ad) 

    ad_mlp_output = utils.mlp(input=ad_mlp_input,params=params,layer_list = [256],op_name = "ad_1")
    ad_mlp_output = tf.concat([ad_mlp_output,main_bias_1],-1)
    ad_mlp_output = utils.mlp(input=ad_mlp_output,params=params, layer_list = [128],op_name = "ad_2")
    ad_mlp_output = tf.concat([ad_mlp_output,main_bias_2],-1)

    logits = tf.squeeze(tf.layers.dense(ad_mlp_output, 1, name='pred_ad')) 
    preds = tf.nn.sigmoid(logits)
    main_mlp_output = tf.boolean_mask(main_mlp_output,self.mask_main)
    main_logits = tf.squeeze(tf.layers.dense(main_mlp_output, 1, name='pred_main')) 
    main_preds = tf.nn.sigmoid(main_logits)
    
    self.main_logits = main_logits
    self.main_preds = main_preds

    self.logits = logits
    self.preds = preds

    

  def build_embedding_layer(self, params):
    self.main_sid_embeddings = utils.build_embeddings(
      emb_name='main_sku_id_embeddings',
      vocab_size=params.num_skus//2,
      emb_dim=params.emb_dim
    )
    self.main_cid_embeddings = utils.build_embeddings(
      emb_name='main_cate_id_embeddings',
      vocab_size=params.num_cates//2,
      emb_dim=params.emb_dim
    )
    self.main_vid_embeddings = utils.build_embeddings(
      emb_name='main_vender_id_embeddings',
      vocab_size=params.num_venders//2,
      emb_dim=params.emb_dim
    )
    self.main_bid_embeddings = utils.build_embeddings(
      emb_name='main_brand_id_embeddings',
      vocab_size=params.num_brands//2,
      emb_dim=params.emb_dim
    )
    self.main_qid_embeddings = utils.build_embeddings(
      emb_name='main_query_embeddings',
      vocab_size=params.num_queries//2,
      emb_dim=params.emb_dim
    )
    self.main_uid_embeddings = utils.build_embeddings(
      emb_name='main_user_embeddings',
      vocab_size=params.num_users//2,
      emb_dim=params.emb_dim
    )
    self.main_price_embeddings = utils.build_embeddings(
      emb_name='main_price_embeddings',
      vocab_size=params.num_prices//2,
      emb_dim=params.emb_dim
    )

    self.ad_sid_embeddings = utils.build_embeddings(
      emb_name='ad_sku_id_embeddings',
      vocab_size=params.num_skus//2,
      emb_dim=params.emb_dim
    )
    self.ad_cid_embeddings = utils.build_embeddings(
      emb_name='ad_cate_id_embeddings',
      vocab_size=params.num_cates//2,
      emb_dim=params.emb_dim
    )
    self.ad_vid_embeddings = utils.build_embeddings(
      emb_name='ad_vender_id_embeddings',
      vocab_size=params.num_venders//2,
      emb_dim=params.emb_dim
    )
    self.ad_bid_embeddings = utils.build_embeddings(
      emb_name='ad_brand_id_embeddings',
      vocab_size=params.num_brands//2,
      emb_dim=params.emb_dim
    )
    self.ad_qid_embeddings = utils.build_embeddings(
      emb_name='ad_query_embeddings',
      vocab_size=params.num_queries//2,
      emb_dim=params.emb_dim
    )
    self.ad_uid_embeddings = utils.build_embeddings(
      emb_name='ad_user_embeddings',
      vocab_size=params.num_users//2,
      emb_dim=params.emb_dim
    )
    self.ad_price_embeddings = utils.build_embeddings(
      emb_name='ad_price_embeddings',
      vocab_size=params.num_prices//2,
      emb_dim=params.emb_dim
    )

    
  def build_input_layer(self, inputs, params):
    main_sid_emb = tf.nn.embedding_lookup(self.main_sid_embeddings, inputs['sid']%(params.num_skus//2))
    main_vid_emb = tf.nn.embedding_lookup(self.main_vid_embeddings, inputs['vid']%(params.num_venders//2))
    main_cid_emb = tf.nn.embedding_lookup(self.main_cid_embeddings, inputs['cid']%(params.num_cates//2))
    main_bid_emb = tf.nn.embedding_lookup(self.main_bid_embeddings, inputs['bid']%(params.num_brands//2))
    main_price_emb = tf.nn.embedding_lookup(self.main_price_embeddings, inputs['price']%(params.num_prices//2))


    main_item_emb = tf.concat([main_sid_emb, main_vid_emb, main_cid_emb, main_bid_emb, main_price_emb], axis=1)
    self.main_item_emb = tf.reshape(main_item_emb, [-1, params.emb_dim * 5])

    main_qid_emb = tf.nn.embedding_lookup(self.main_qid_embeddings, inputs['qid']%(params.num_queries//2))
    self.main_query_emb = tf.reshape(main_qid_emb, [-1, params.emb_dim])

    main_uid_emb = tf.nn.embedding_lookup(self.main_uid_embeddings, inputs['uid']%(params.num_users//2))
    self.main_user_id_emb = tf.reshape(main_uid_emb, [-1, params.emb_dim])
    self.main_user_emb = self.main_user_id_emb

    ad_sid_emb = tf.nn.embedding_lookup(self.ad_sid_embeddings, inputs['sid']%(params.num_skus//2))
    ad_vid_emb = tf.nn.embedding_lookup(self.ad_vid_embeddings, inputs['vid']%(params.num_venders//2))
    ad_cid_emb = tf.nn.embedding_lookup(self.ad_cid_embeddings, inputs['cid']%(params.num_cates//2))
    ad_bid_emb = tf.nn.embedding_lookup(self.ad_bid_embeddings, inputs['bid']%(params.num_brands//2))
    ad_price_emb = tf.nn.embedding_lookup(self.ad_price_embeddings, inputs['price']%(params.num_prices//2))


    ad_item_emb = tf.concat([ad_sid_emb, ad_vid_emb, ad_cid_emb, ad_bid_emb, ad_price_emb], axis=1)
    self.ad_item_emb = tf.reshape(ad_item_emb, [-1, params.emb_dim * 5])

    ad_qid_emb = tf.nn.embedding_lookup(self.ad_qid_embeddings, inputs['qid']%(params.num_queries//2))
    self.ad_query_emb = tf.reshape(ad_qid_emb, [-1, params.emb_dim])

    ad_uid_emb = tf.nn.embedding_lookup(self.ad_uid_embeddings, inputs['uid']%(params.num_users//2))
    self.ad_user_id_emb = tf.reshape(ad_uid_emb, [-1, params.emb_dim])
    self.ad_user_emb = self.ad_user_id_emb
    self.sid = inputs['sid']
    self.global_group_id = inputs['global_group_id']


  def build_loss_layer(self, params):
    self.logits = tf.reshape(self.logits, [-1, 1])
    self.labels = tf.reshape(self.labels, [-1, 1])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels))

    self.main_logits = tf.reshape(self.main_logits, [-1, 1])
    self.main_labels = tf.reshape(self.main_labels, [-1, 1])
    main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.main_logits, labels=self.main_labels))



    self.loss = loss + main_loss
    self.log_loss = loss
