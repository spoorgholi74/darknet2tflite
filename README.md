# YOLOv3 dark2lite implementation
- [Adopted from](https://github.com/xiaochus/YOLOv3). 

## Requirement
- OpenCV 3.4
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3

## Quick start

- Copy your weight files and the config file in the root directory.

- The following command will translate the model to keras .h5 model.
```
python yad2k.py cfg\yolo.cfg yolov3.weights data\yolo.h5
```
- Copy the .h5 model to the data/ directory to be able to run the demo.

- run follow command to show the keras demo. The result can be found in `images\res\` floder.
```
python demo_keras.py
```
- run follow command to show the tflite demo.
```
python demo_tflite.py
```
