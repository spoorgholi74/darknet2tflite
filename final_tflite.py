

import os
import cv2
import time
import numpy as np
import tensorflow as tf
from model.yolo_model import YOLO


TFLITE_MODEL = 'yolo3_70_pruned.tflite'
input_file = os.path.join('images/test/','1.jpg')


tflite_interpreter = tf.lite.Interpreter(TFLITE_MODEL)
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

'''
print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details =="+str(len(output_details)))
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])
print("name:", output_details[1]['name'])
print("shape:", output_details[1]['shape'])
print("type:", output_details[1]['dtype'])
print("name:", output_details[2]['name'])
print("shape:", output_details[2]['shape'])
print("type:", output_details[2]['dtype'])
'''


image = cv2.imread(input_file)
cv2.imshow('final', image)
#cv2.waitKey(0)

def process_image(img):
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
converted_image = process_image(image);

tflite_interpreter.allocate_tensors()

tflite_interpreter.set_tensor(input_details[0]['index'], converted_image)
tflite_interpreter.invoke()

def rs(ten):
  a = list(ten.shape)
  a = tuple(a)
  print(a)
  return ten.reshape(a)

tflite_model_predictions = [
                            rs(tflite_interpreter.get_tensor(output_details[0]['index'])),
                            rs(tflite_interpreter.get_tensor(output_details[1]['index'])),
                            rs(tflite_interpreter.get_tensor(output_details[2]['index']))]

y = YOLO(0.6, 0.5)
w, h , c= image.shape

start = time.time()
boxes, classes, scores = y._yolo_out(tflite_model_predictions, (w,h))
end = time.time()
print('time: {0:.2f}s'.format(end - start))

all_classes = load_labels('data/coco_classes.txt')

def draw(image, boxes, scores, classes, all_classes):
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print(box)

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()

draw(image, boxes, scores, classes, all_classes)
cv2.imwrite('detected.jpg', image)
cv2.imshow('detected!', image)
cv2.waitKey(0)