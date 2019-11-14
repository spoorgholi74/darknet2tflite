import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file( 'data/yolo.h5' ) # Your model's name
model = converter.convert()
file = open( 'yolov3.tflite' , 'wb' ) 
file.write( model )