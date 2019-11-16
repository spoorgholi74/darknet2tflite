import tensorflow as tf

'''
new_model= tf.keras.models.load_model(filepath="data/yolo.h5")

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = tflite_converter.convert()
open("test1.tflite", "wb").write(tflite_model)
'''

converter = tf.lite.TFLiteConverter.from_keras_model_file( 'data/yolo.h5' ) # Your model's name
model = converter.convert()
file = open( 'yolov3.tflite' , 'wb' ) 
file.write( model )
