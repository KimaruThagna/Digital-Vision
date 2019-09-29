from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet() # others include SqueezeNet, ResNet, InceptionV3 and DenseNet
model_trainer.setDataDirectory("idenprof") # dataset
model_trainer.trainModel(num_objects=10, # prediction classes
                         num_experiments=200,
                         enhance_data=True, # allow data augmentation
                         batch_size=32, # images per cycle
                         show_network_summary=True)