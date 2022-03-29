# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 Northern System Service Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
r"""ShapeMask で物体検出を行い、マスク画像を作成するツール

与えられた画像に対して、ShapeMask で物体検出を行う。
検出したオブジェクトごとに1枚のバイナリマスク画像を書き出す。

python /path/to/inference_via.py \
    --image_size 256 \
    --model=shapemask \
    --checkpoint_path=/path/to/model.ckpt-30000 \
    --config_file=/path/to/config.yaml \
    --image_file_pattern='/path/to/images/*.JPG' \
    --label_map_file=/path/to/labels.csv \
    --export_to=/path/to/reslt/

マスク画像名のフォーマットは下記の通り。
    {元ファイル名}_{オブジェクト連番}_conf{確信度}_label{ラベル名}.png
    例) DSC0001_03_conf0.98_labelfruit.png
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io
import os
from tqdm import tqdm
from pathlib import Path

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from configs import factory as config_factory
from dataloader import mode_keys
from modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', 'retinanet', 'Support `retinanet`, `mask_rcnn` and `shapemask`.')
flags.DEFINE_integer('image_size', 256, 'The image size.')
flags.DEFINE_string(
    'checkpoint_path', '', 'The path to the checkpoint file.')
flags.DEFINE_string(
    'config_file', '', 'The config file template.')
flags.DEFINE_string(
    'params_override', '', 'The YAML file/string that specifies the parameters '
    'override in addition to the `config_file`.')
flags.DEFINE_string(
    'label_map_file', '',
    'The label map file. See --label_map_format for the definition.')
flags.DEFINE_string(
    'label_map_format', 'csv',
    'The format of the label map file. Currently only support `csv` where the '
    'format of each row is: `id:name`.')
flags.DEFINE_string(
    'image_file_pattern', '',
    'The glob that specifies the image file pattern.')
flags.DEFINE_string(
    'export_to', '/path/to/export/to',
    'The directory to export masks to.')


def main(unused_argv):
  del unused_argv

  export_to = Path(FLAGS.export_to)
  if not export_to.is_dir():
    export_to.mkdir()

  # Load the label map.
  print(' - Loading the label map...')
  label_map_dict = {}
  if FLAGS.label_map_format == 'csv':
    with tf.gfile.Open(FLAGS.label_map_file, 'r') as csv_file:
      reader = csv.reader(csv_file, delimiter=':')
      for row in reader:
        if len(row) != 2:
          raise ValueError('Each row of the csv label map file must be in '
                           '`id:name` format.')
        id_index = int(row[0])
        name = row[1]
        label_map_dict[id_index] = {
            'id': id_index,
            'name': name,
        }
  else:
    raise ValueError(
        'Unsupported label map format: {}.'.format(FLAGS.label_mape_format))

  params = config_factory.config_generator(FLAGS.model)
  if FLAGS.config_file:
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.validate()
  params.lock()

  model = model_factory.model_generator(params)

  with tf.Graph().as_default():
    image_input = tf.placeholder(shape=(), dtype=tf.string)
    image = tf.io.decode_image(image_input, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [FLAGS.image_size, FLAGS.image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])

    # batching.
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    # model inference
    outputs = model.build_outputs(
        images, {'image_info': images_info}, mode=mode_keys.PREDICT)

    outputs['detection_boxes'] = (
        outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

    predictions = outputs

    # Create a saver in order to load the pre-trained checkpoint.
    saver = tf.train.Saver()

    with tf.Session() as sess:
      print(' - Loading the checkpoint...')
      saver.restore(sess, FLAGS.checkpoint_path)

      image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
      for i, image_file in tqdm(
          enumerate(image_files), ascii=True, total=len(image_files)):
        logging.debug(
          ' - Generating masks for {} ({})...'.format(i, image_file))

        logging.debug(' - Opening {}...'.format(image_file))
        with tf.gfile.GFile(image_file, 'rb') as f:
          image_bytes = f.read()

        image = Image.open(image_file)
        image = image.convert('RGB')  # needed for images with 4 channels.
        width, height = image.size
        logging.debug(' - Image size is {}.'.format(image.size))
        np_image = (
          np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8))

        predictions_np = sess.run(
          predictions, feed_dict={image_input: image_bytes})

        num_detections = int(predictions_np['num_detections'][0])
        np_boxes = predictions_np['detection_boxes'][0, :num_detections]
        np_scores = predictions_np['detection_scores'][0, :num_detections]
        np_classes = predictions_np['detection_classes'][0, :num_detections]
        np_classes = np_classes.astype(np.int32)
        np_masks = None
        if 'detection_masks' in predictions_np:
          instance_masks = predictions_np['detection_masks'][0, :num_detections]
          np_masks = mask_utils.paste_instance_masks(
            instance_masks, box_utils.yxyx_to_xywh(np_boxes), height, width)
        # np_masks is a numpy array, shape==(n, H, W)
        
        ## Export masks
        mask_basename = Path(image_file).stem

        for i, np_mask in enumerate(np_masks):
          fname = \
            '{basename}_{i:02d}_conf{score:.2f}_label{label:s}.png'.format(
              basename=mask_basename, i=i, score=np_scores[i], 
              label=label_map_dict[np_classes[i]]['name']
            )
          mask_path = Path(FLAGS.export_to)/fname
          logging.debug('Exporting {}'.format(mask_path))
          im = Image.fromarray(np_mask)
          im.save(mask_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('model')
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('image_file_pattern')
  flags.mark_flag_as_required('export_to')
  logging.set_verbosity(logging.WARNING)
  tf.app.run(main)
