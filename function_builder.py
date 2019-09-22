"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
import modeling
import xlnet


def get_ner_loss(FLAGS,features, is_training, num_labels):
    inp = tf.transpose(features["input_ids"], [1, 0])
    seg_id = tf.transpose(features["segment_ids"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])
    label_id = tf.transpose(features["label_ids"], [1, 0])

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)
    output_layer = xlnet_model.get_sequence_output()

    def hidden2tag(hiddenlayer,numclass):
        linear = tf.keras.layers.Dense(numclass,activation=None)
        return linear(hiddenlayer)


    def softmax_layer(logits, labels, num_labels, mask):
        logits = tf.reshape(logits, [-1, num_labels])
        labels = tf.reshape(labels, [-1])
        mask = tf.cast(mask, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
        loss *= tf.reshape(mask, [-1])
        loss = tf.reduce_sum(loss)
        total_size = tf.reduce_sum(mask)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        loss /= total_size
        # predict not mask we could filtered it in the prediction part.
        probabilities = tf.math.softmax(logits, axis=-1)
        predict = tf.math.argmax(probabilities, axis=-1)
        return loss, predict
    

    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer,num_labels)
    logits = tf.reshape(logits,[-1,128,num_labels])
    loss, predict = softmax_layer(logits,label_id,num_labels,inp_mask)
      
      
    return loss, logits, predict