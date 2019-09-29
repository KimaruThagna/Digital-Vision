from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3() # others can include RetinaNet
trainer.setDataDirectory(data_directory="inputs/hololens")
trainer.setTrainConfig(object_names_array=["hololens"], # an array of names of all objects in your dataset
                       batch_size=4, num_experiments=100,
                       train_from_pretrained_model="models/pretrained-yolov3.h5")# use transfer learning from a pretrained YOLO model
trainer.trainModel()