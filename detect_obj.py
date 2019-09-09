from imageai.Detection import ObjectDetection,VideoObjectDetection
from glob import glob
import os

'''
TODO
Read upon other models at https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0/
'''
detection_obj = ObjectDetection()
detection_obj.setModelTypeAsRetinaNet()# you can choose between retinaNet, YOLOv3 and TinyYOLOv3
detection_obj.setModelPath( "models/resnet50_coco_best_v2.0.1.h5")
detection_obj.loadModel()
for img in glob("inputs/*"):
    detections = detection_obj.detectObjectsFromImage(input_image= img,
                                                      output_image_path=os.path.join("outputs/" , f'detected_{str(img).split("/")[1]}'))
    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )