'''
After running the training file, object_detection_custom_training.py,
Run this file to evaluate the generated model from training
'''
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.evaluateModel(model_path="hololens/models", # where model is
                      json_path="hololens/json/detection_config.json",
                      iou_threshold=0.5, # minimum desired inersect over union of the Mean Average Precision
                      object_threshold=0.3, #desired minimum class score for the mAP computation
                      nms_threshold=0.5)#desired Non-maximum suppression for the mAP computation.
