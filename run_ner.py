from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from absl import flags, logging
import os
import pickle
import sys
import csv
import collections
import numpy as np
import time
import math
import json
import random
from copy import copy
from collections import defaultdict as dd

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import sentencepiece as spm
import metrics
from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
import function_builder
from classifier_utils import PaddingInputExample
# from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids, printable_text

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="",
                    help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# training
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=1000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
                     help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_float("predict_threshold", default=0,
                   help="Threshold for binary prediction.")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=128,
                     help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128,
                     help="batch size for prediction.")
flags.DEFINE_string("predict_dir", default=None,
                    help="Dir for saving prediction files.")
flags.DEFINE_bool("eval_all_ckpt", default=False,
                  help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("predict_ckpt", default=None,
                    help="Ckpt path for do_predict. If None, use the last one.")

# task specific
flags.DEFINE_string("task_name", default=None, help="Task name")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1,
                     help="Num passes for processing training data. "
                          "This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None,
                    help="Classifier layer scope.")
flags.DEFINE_bool("is_regression", default=False,
                  help="Whether it's a regression task.")

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        rf = open(input_file, 'r')
        lines = [];
        words = [];
        labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if len(line.strip()) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l, w))
                words = []
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test_part.txt")), "test")

    def get_labels(self):
        """See base class."""
        # return ["[PAD]", "o", "B-AS", "I-AS", "B-OP", "I-OP", "X", "[CLS]", "[SEP]"]
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return ["[PAD]","O","B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = preprocess_text(line[1])
            labels = preprocess_text(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenize_fn):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.output_dir + "/label2id.pkl", "wb") as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(" ")
    labellist = example.label.split(" ")
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # print(word)
        token = [preprocess_text(word, lower=True)]
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    input_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    # print('tokens')
    # print(tokens)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    #ntokens.append("[SEP]")
    #segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    #label_ids.append(label_map["[SEP]"])
    for i in ntokens:
        input_ids.append(tokenize_fn(i))
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append(0)
        label_ids.append(0)
    # print(len(input_mask))
    # print(len(input_ids))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    # print('input_ids')
    # print(input_ids)
    # print(type(input_ids))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenize_fn, output_file,
        num_passes=1):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenize_fn)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_float_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def get_model_fn(num_labels):
    def model_fn(features, labels, mode, params):

        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        total_loss, logits, predicts = function_builder.get_ner_loss(FLAGS, features, is_training, num_labels)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        if mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits, num_labels, mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels - 1, weights=mask)
                return {
                    "confusion_matrix": cm
                }
                #

            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
            return eval_spec

        elif mode == tf.estimator.ModeKeys.PREDICT:
            #label_ids = tf.reshape(features["label_ids"], [-1])
            # predictions = {
            #     "predicts": predicts,
            # }
            #print('predicts')
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn)
            return output_spec

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### Constucting training TPUEstimatorSpec with new cache.
        #train_spec = tf.estimator.EstimatorSpec(
        #    mode=mode, loss=total_loss, train_op=train_op)
        train_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn)


        return train_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token != "[PAD]" and token != "[CLS]" and true_l != "X":
        #
        if predict == "X" and not predict.startswith("##"):
            predict = "O"
        line = "{}\t{}\t{}\n".format(token, true_l, predict)
        wf.write(line)


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    with open(output_predict_file, 'w') as wf:
        for i, prediction in enumerate(result):
            _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if FLAGS.do_predict:
        predict_dir = FLAGS.predict_dir
        if not tf.gfile.Exists(predict_dir):
            tf.gfile.MakeDirs(predict_dir)

    processors = {
        "ner": NerProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval, `do_predict` or "
            "`do_submit` must be True.")

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels() if not FLAGS.is_regression else None
    print('label_list')
    print(label_list)

    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        if sp.PieceToId(text) == 0:
            return 99999
        return sp.PieceToId(text)

    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn(len(label_list))

    spm_basename = os.path.basename(FLAGS.spiece_model_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    #if FLAGS.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    #else:
    #    estimator = tf.estimator.Estimator(
    #        model_fn=model_fn,
    #        config=run_config)

    if FLAGS.do_train:
        train_file_base = "{}.len-{}.train.tf_record".format(
            spm_basename, FLAGS.max_seq_length)
        train_file = os.path.join(FLAGS.output_dir, train_file_base)
        tf.logging.info("Use tfrecord file {}".format(train_file))

        # print('get train examples')
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        np.random.shuffle(train_examples)
        tf.logging.info("Num of train samples: {}".format(len(train_examples)))

        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            train_file, FLAGS.num_passes)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    # if FLAGS.do_eval or FLAGS.do_predict:
    #     if FLAGS.eval_split == "dev":
    #         eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    #     else:
    #         eval_examples = processor.get_test_examples(FLAGS.data_dir)
    #
    #     tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            eval_file)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as wf:
            logging.info("***** Eval results *****")
            confusion_matrix = result["confusion_matrix"]
            p, r, f = metrics.calculate(confusion_matrix, len(label_list) - 1)
            logging.info("***********************************************")
            logging.info("********************P = %s*********************", str(p))
            logging.info("********************R = %s*********************", str(r))
            logging.info("********************F = %s*********************", str(f))
            logging.info("***********************************************")

    if FLAGS.do_predict:
        # eval_file_base = "{}.len-{}.{}.predict.tf_record".format(
        #     spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        # eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

        test_file_base = "{}.len-{}.predict.tf_record".format(
            spm_basename, FLAGS.max_seq_length)
        test_file = os.path.join(FLAGS.output_dir, test_file_base)
        test_examples = processor.get_test_examples(FLAGS.data_dir)

        file_based_convert_examples_to_features(
            test_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            test_file)

        pred_input_fn = file_based_input_fn_builder(
            input_file=test_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=FLAGS.train_batch_size)

        predict_results = []

        result = estimator.predict(input_fn=pred_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # print(result)
        print(list(result))

        # print(result['predicts'])
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        # Writer(output_predict_file, result, batch_tokens, batch_labels, id2label)


if __name__ == "__main__":
    tf.app.run()