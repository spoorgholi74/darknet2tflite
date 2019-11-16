# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from PIL import Image
from model.yolo_model import YOLO
from demo import *
import cv2

import tensorflow as tf # TF2


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    #cv2.imshow('before resize', img)
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    #cv2.imshow('after expand', image)
    image = np.expand_dims(image, axis=0)
    #cv2.waitKey(0)

    return image



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '-i',
      '--image',
      default='/tmp/grace_hopper.bmp',
      help='image to be classified')
    parser.add_argument(
      '-m',
      '--model_file',
      default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
      help='.tflite model to be executed')
    parser.add_argument(
      '-l',
      '--label_file',
      default='/tmp/labels.txt',
      help='name of file containing labels')
    parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
    parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    pimage = process_image(image)

    interpreter = tf.lite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('output details = ', output_details)

    '''
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print(height, ' * ', width)
    image = Image.open(args.image)
    print('image size is ', image.size)
    img = image.resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)
    print('Shape is ', input_data.shape)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std


    interpreter.set_tensor(input_details[0]['index'], input_data)

    '''
    interpreter.set_tensor(input_details[0]['index'], pimage)
    interpreter.invoke()


    def rs(ten):
        a = list(ten.shape)
        a = a[:-1]
        a += [3, 7]
        a = tuple(a)
        print(a)
        return ten.reshape(a)

    output_data = [
                    rs(interpreter.get_tensor(output_details[0]['index'])),
                    rs(interpreter.get_tensor(output_details[1]['index'])),
                    rs(interpreter.get_tensor(output_details[2]['index']))]
    

    #output_data = interpreter.get_tensor(output_details[0]['index'])
    print('output_data shape is ', output_data.shape)
    print('output data is ', output_data)
    #results = np.squeeze(output_data)

    #top_k = results.argsort()[-5:][::-1]
    #print('top_k = ', top_k)
    labels = load_labels(args.label_file)
    print(labels)

    
    yolo = YOLO(0.6, 0.5)
    boxes, classes, scores = yolo.predict(input_data, image.size, outs = output_data)
    #print(boxes)

    if boxes is not None:
        draw(image, boxes, scores, classes, labels)

    image = image.save('detected.jpg')
    '''

    
    for i in top_k:
        if floating_model:
            print('i=', i)
            print('reslut[i] = ', results[i])
            print('labels[i] = ', labels[i])
            #print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

    '''